"""
Метрики  mAP, Precision,  Recall , средний IoU при IoU≥порога
Классы цифр (метка 10 в SVHN = цифра 0)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torchvision.transforms.functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou


def label_to_digit_class(t: torch.Tensor) -> torch.Tensor:
    return (t.long() % 10).long()


@torch.no_grad()
def match_image(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    tgt_boxes: torch.Tensor,
    tgt_labels: torch.Tensor,
    iou_thresh: float,
    score_thresh: float,
) -> Tuple[int, int, int, List[float]]:
    """
    Жадное сопоставление по убыванию score.
    Возвращает tp, fp, fn, список IoU для TP.
    """
    m = pred_scores >= score_thresh
    pred_boxes = pred_boxes[m]
    pred_labels = label_to_digit_class(pred_labels[m])
    pred_scores = pred_scores[m]

    tgt_labels = label_to_digit_class(tgt_labels)
    n_p, n_g = len(pred_boxes), len(tgt_boxes)
    if n_g == 0:
        return 0, n_p, 0, []

    gt_matched = torch.zeros(n_g, dtype=torch.bool, device=tgt_boxes.device)
    tp_list: List[float] = []

    if n_p == 0:
        return 0, 0, n_g, []

    order = torch.argsort(pred_scores, descending=True)
    tp = fp = 0
    for idx in order:
        pb = pred_boxes[idx : idx + 1]
        pl = pred_labels[idx]
        ious = box_iou(pb, tgt_boxes)[0]
        best_iou, best_j = 0.0, -1
        for j in range(n_g):
            if gt_matched[j]:
                continue
            if tgt_labels[j] != pl:
                continue
            v = float(ious[j].item())
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thresh:
            gt_matched[best_j] = True
            tp += 1
            tp_list.append(best_iou)
        else:
            fp += 1

    fn = int((~gt_matched).sum().item())
    return tp, fp, fn, tp_list


@torch.no_grad()
def evaluate_svhn_test(
    model,
    dataset,
    device: torch.device,
    score_thresh: float = 0.5,
    iou_thresh_pr: float = 0.5,
) -> Dict[str, float]:
    model.eval()
    map_metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)

    tp_t = fp_t = fn_t = 0
    all_tp_ious: List[float] = []

    for i in range(len(dataset)):
        pil, tgt = dataset[i]
        img = F.to_tensor(pil).to(device)
        out = model([img])[0]
        keep = (out["scores"] >= score_thresh) & (out["labels"] > 0)
        pb = out["boxes"][keep].cpu()
        pl = out["labels"][keep].cpu()
        ps = out["scores"][keep].cpu()
        tb = tgt["boxes"].cpu()
        tl = tgt["labels"].cpu()

        pbc = label_to_digit_class(pl)
        tbc = label_to_digit_class(tl)
        map_metric.update(
            [dict(boxes=pb, scores=ps, labels=pbc)],
            [dict(boxes=tb, labels=tbc)],
        )

        tp, fp, fn, ious = match_image(pb, pl, ps, tb, tl, iou_thresh_pr, score_thresh)
        tp_t += tp
        fp_t += fp
        fn_t += fn
        all_tp_ious.extend(ious)

    m = map_metric.compute()
    out_d: Dict[str, float] = {}
    for k in ("map", "map_50", "map_75", "mar_100", "mar_10"):
        if k in m:
            out_d[k] = float(m[k])
    denom_p = tp_t + fp_t
    denom_r = tp_t + fn_t
    precision = float(tp_t / denom_p) if denom_p > 0 else 0.0
    recall = float(tp_t / denom_r) if denom_r > 0 else 0.0
    mean_iou_tp = float(sum(all_tp_ious) / len(all_tp_ious)) if all_tp_ious else 0.0

    out_d["precision_iou50"] = precision
    out_d["recall_iou50"] = recall
    out_d["mean_iou_matched_tp"] = mean_iou_tp
    out_d["tp"] = float(tp_t)
    out_d["fp"] = float(fp_t)
    out_d["fn"] = float(fn_t)
    return out_d
