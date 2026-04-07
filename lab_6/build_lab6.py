#!/usr/bin/env python3
"""Генерирует lab_6.ipynb с пояснениями перед каждым код-блоком."""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(text: str):
    return new_markdown_cell(text)


def code(text: str):
    return new_code_cell(text)


ROOT = Path(__file__).resolve().parent
cells = []

cells.append(
    md(
        """# Лабораторная 6

Здесь делаем сегментацию дорожных знаков на `Mask R-CNN` и считаем метрики так, как просили в задании.
Код подготовлен так, чтобы на Windows с NVIDIA (RTX 5070) приоритетно запускаться на CUDA.
"""
    )
)

cells.append(
    md(
        """Сначала импортируем библиотеки, задаём seed для воспроизводимости и аккуратно определяем пути.
Плюс сразу выбираем устройство: `cuda` если доступно, потом `mps`, иначе `cpu`.
"""
    )
)
cells.append(
    code(
        """import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path.cwd()
for cand in (ROOT, ROOT / "lab_6", ROOT.parent / "lab_6"):
    if (cand / "sign_dataset").is_dir():
        ROOT = cand
        break

TRAIN_DIR = ROOT / "sign_dataset" / "train"
VAL_DIR = ROOT / "sign_dataset" / "val"
assert TRAIN_DIR.is_dir(), TRAIN_DIR
assert VAL_DIR.is_dir(), VAL_DIR

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("ROOT:", ROOT.resolve())
print("TRAIN_DIR:", TRAIN_DIR)
print("VAL_DIR:", VAL_DIR)
print("device:", device)
print("torch:", torch.__version__)
if device.type == "cuda":
    print("cuda:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))"""
    )
)

cells.append(
    md(
        """Смотрим размер train/val и какие именно 8 классов реально есть в разметке.
Также строим remap классов в непрерывный диапазон для `Mask R-CNN`.
"""
    )
)
cells.append(
    code(
        """def list_image_ids(folder: Path):
    ids = []
    for p in folder.glob("*.jpg"):
        if p.stem.isdigit() and (folder / f"{p.stem}.jpg_coco.json").is_file():
            ids.append(int(p.stem))
    return sorted(ids)


train_ids = list_image_ids(TRAIN_DIR)
val_ids = list_image_ids(VAL_DIR)
print("train images:", len(train_ids))
print("val images:", len(val_ids))


def collect_class_ids(folder: Path, ids):
    out = set()
    for i in ids:
        d = json.loads((folder / f"{i}.jpg_coco.json").read_text())
        out.update(d.get("class_ids", []))
    return sorted(out)


orig_class_ids = sorted(set(collect_class_ids(TRAIN_DIR, train_ids)) | set(collect_class_ids(VAL_DIR, val_ids)))
class_to_new = {cid: idx + 1 for idx, cid in enumerate(orig_class_ids)}
new_to_class = {v: k for k, v in class_to_new.items()}

print("оригинальные class_ids:", orig_class_ids)
print("remap ->", class_to_new)"""
    )
)

cells.append(
    md(
        """Делаем класс датасета: читаем `jpg` и `jpg_coco.json`, достаём боксы/маски/метки.
Формат приводим к тому, который ожидает `torchvision` для instance segmentation.
"""
    )
)
cells.append(
    code(
        """class RoadSignsMaskDataset(Dataset):
    def __init__(self, folder: Path, image_ids, class_to_new_map):
        self.folder = Path(folder)
        self.ids = list(image_ids)
        self.class_to_new = class_to_new_map

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_path = self.folder / f"{image_id}.jpg"
        ann_path = self.folder / f"{image_id}.jpg_coco.json"

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        ann = json.loads(ann_path.read_text())

        rois = ann.get("rois", [])
        class_ids = ann.get("class_ids", [])
        masks_raw = np.asarray(ann.get("masks", []), dtype=np.uint8)

        if masks_raw.size == 0 or len(rois) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, h, w), dtype=torch.uint8)
        else:
            if masks_raw.ndim == 3:
                masks = torch.as_tensor(masks_raw.transpose(2, 0, 1), dtype=torch.uint8)
            else:
                masks = torch.zeros((0, h, w), dtype=torch.uint8)

            boxes_list, labels_list = [], []
            for i, roi in enumerate(rois):
                y1, x1, y2, x2 = roi
                boxes_list.append([x1, y1, x2, y2])
                labels_list.append(self.class_to_new[int(class_ids[i])])

            boxes = torch.as_tensor(boxes_list, dtype=torch.float32)
            labels = torch.as_tensor(labels_list, dtype=torch.int64)
            boxes[:, 0::2].clamp_(0, w)
            boxes[:, 1::2].clamp_(0, h)

            if masks.shape[0] != boxes.shape[0]:
                n = min(masks.shape[0], boxes.shape[0])
                masks = masks[:n]
                boxes = boxes[:n]
                labels = labels[:n]

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([image_id]),
        }
        return F.to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))"""
    )
)

cells.append(
    md(
        """Инициализируем `Mask R-CNN ResNet50-FPN` с предобученными весами и меняем головы под наши 8 классов.
Параметры обучения тоже задаём здесь (batch size, оптимизатор, scheduler).
"""
    )
)
cells.append(
    code(
        """train_ds = RoadSignsMaskDataset(TRAIN_DIR, train_ids, class_to_new)
val_ds = RoadSignsMaskDataset(VAL_DIR, val_ids, class_to_new)

BATCH_SIZE = 8 if device.type == "cuda" else 2
NUM_WORKERS = 4 if device.type == "cuda" else 0
PIN_MEMORY = device.type == "cuda"

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    collate_fn=collate_fn,
)

NUM_CLASSES = 1 + len(orig_class_ids)  # фон + 8 классов

weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, NUM_CLASSES)

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

print("NUM_CLASSES:", NUM_CLASSES)
print("BATCH_SIZE:", BATCH_SIZE)
print("NUM_WORKERS:", NUM_WORKERS)
print("PIN_MEMORY:", PIN_MEMORY)"""
    )
)

cells.append(
    md(
        """Запускаем обучение.
Если мало памяти на GPU, просто уменьшите `BATCH_SIZE` до 2.
"""
    )
)
cells.append(
    code(
        """def train_one_epoch(model, loader, optimizer, device):
    model.train()
    losses_all = []
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for images, targets in loader:
        images = [img.to(device, non_blocking=use_amp) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss_dict = model(images, targets)
            loss = sum(v for v in loss_dict.values())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses_all.append(float(loss.detach().cpu()))

    return float(np.mean(losses_all)) if losses_all else 0.0


EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    mean_loss = train_one_epoch(model, train_loader, optimizer, device)
    lr_scheduler.step()
    print(f"epoch {epoch}/{EPOCHS}, mean loss = {mean_loss:.4f}")

weights_path = ROOT / "maskrcnn_signs.pth"
torch.save(model.state_dict(), weights_path)
print("weights saved:", weights_path)"""
    )
)

cells.append(
    md(
        """Ниже функции для расчёта метрик сегментации:
`IoU`, `Precision`, `Recall`, `L2`, плюс доли изображений с IoU выше порогов 0.5 / 0.75 / 0.9.
"""
    )
)
cells.append(
    code(
        """def binary_iou(pred_mask, true_mask, eps=1e-9):
    inter = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return 1.0
    return (inter + eps) / (union + eps)


def precision_recall(pred_mask, true_mask, eps=1e-9):
    tp = np.logical_and(pred_mask, true_mask).sum()
    fp = np.logical_and(pred_mask, np.logical_not(true_mask)).sum()
    fn = np.logical_and(np.logical_not(pred_mask), true_mask).sum()
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return precision, recall


def l2_distance(pred_mask, true_mask):
    a = pred_mask.astype(np.float32)
    b = true_mask.astype(np.float32)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def aggregate_masks(mask_arr):
    if mask_arr.ndim == 3:
        return (mask_arr > 0).any(axis=0)
    if mask_arr.ndim == 2:
        return mask_arr > 0
    return np.zeros((1, 1), dtype=bool)


@torch.no_grad()
def evaluate_segmentation(model, loader, device, score_thr=0.5, mask_thr=0.5):
    model.eval()
    ious, precisions, recalls, l2s = [], [], [], []
    hits_05, hits_075, hits_09, total = 0, 0, 0, 0

    for images, targets in loader:
        image = images[0].to(device)
        target = targets[0]
        out = model([image])[0]
        keep = out["scores"].detach().cpu().numpy() >= score_thr

        if out["masks"].shape[0] > 0:
            pred_masks = out["masks"][keep].detach().cpu().numpy()[:, 0, :, :] >= mask_thr
            pred_union = aggregate_masks(pred_masks)
        else:
            h, w = image.shape[1], image.shape[2]
            pred_union = np.zeros((h, w), dtype=bool)

        if target["masks"].shape[0] > 0:
            gt_masks = target["masks"].detach().cpu().numpy() > 0
            gt_union = aggregate_masks(gt_masks)
        else:
            gt_union = np.zeros_like(pred_union, dtype=bool)

        iou = binary_iou(pred_union, gt_union)
        p, r = precision_recall(pred_union, gt_union)
        l2 = l2_distance(pred_union, gt_union)

        ious.append(iou)
        precisions.append(p)
        recalls.append(r)
        l2s.append(l2)
        hits_05 += int(iou >= 0.5)
        hits_075 += int(iou >= 0.75)
        hits_09 += int(iou >= 0.9)
        total += 1

    return {
        "IoU_mean": float(np.mean(ious)) if total else 0.0,
        "Precision_mean": float(np.mean(precisions)) if total else 0.0,
        "Recall_mean": float(np.mean(recalls)) if total else 0.0,
        "L2_mean": float(np.mean(l2s)) if total else 0.0,
        "IoU>=0.5": hits_05 / total if total else 0.0,
        "IoU>=0.75": hits_075 / total if total else 0.0,
        "IoU>=0.9": hits_09 / total if total else 0.0,
        "N": total,
    }"""
    )
)

cells.append(
    md(
        """Считаем итоговые метрики на `val` и печатаем в удобном виде.
Эти числа можно сразу вставлять в отчёт.
"""
    )
)
cells.append(
    code(
        """val_metrics = evaluate_segmentation(model, val_loader, device=device, score_thr=0.5, mask_thr=0.5)
print("Метрики на val:")
for k, v in val_metrics.items():
    print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")"""
    )
)

cells.append(
    md(
        """Для наглядности показываем несколько картинок из `val` с наложенной предсказанной маской.
Красным цветом отмечена объединённая маска всех найденных знаков.
"""
    )
)
cells.append(
    code(
        """@torch.no_grad()
def show_predictions(model, folder: Path, n=6, score_thr=0.5):
    model.eval()
    ids = list_image_ids(folder)[:n]
    if not ids:
        print("Нет изображений для показа")
        return

    cols = 3
    rows = int(np.ceil(len(ids) / cols))
    plt.figure(figsize=(5 * cols, 4 * rows))

    for i, image_id in enumerate(ids, 1):
        img = Image.open(folder / f"{image_id}.jpg").convert("RGB")
        x = F.to_tensor(img).to(device)
        out = model([x])[0]
        keep = out["scores"].detach().cpu().numpy() >= score_thr

        img_np = np.array(img)
        overlay = img_np.copy()
        if out["masks"].shape[0] > 0 and keep.any():
            pred_masks = out["masks"][keep].detach().cpu().numpy()[:, 0, :, :] >= 0.5
            union = pred_masks.any(axis=0)
            overlay[union] = (0.6 * overlay[union] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)

        plt.subplot(rows, cols, i)
        plt.imshow(overlay)
        plt.title(f"id={image_id}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


show_predictions(model, VAL_DIR, n=6, score_thr=0.5)"""
    )
)

cells.append(
    md(
        """Теперь блок для ваших 10 фото.
Положите изображения в `lab_6/my_photos`. Если есть разметка масок, положите их в `lab_6/my_photos_masks` в формате `name.png` для `name.jpg` (фон 0, знак > 0).
"""
    )
)
cells.append(
    code(
        """MY_PHOTOS_DIR = ROOT / "my_photos"
MY_MASKS_DIR = ROOT / "my_photos_masks"


def list_my_images(folder: Path):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
    return sorted(files)


@torch.no_grad()
def evaluate_my_photos(model, photos_dir: Path, masks_dir: Path | None = None, score_thr=0.5, mask_thr=0.5):
    photos = list_my_images(photos_dir)
    if len(photos) == 0:
        print("В папке my_photos нет изображений")
        return None

    model.eval()
    print("Найдено фото:", len(photos))

    ious, precisions, recalls, l2s = [], [], [], []
    hits_05, hits_075, hits_09, gt_count = 0, 0, 0, 0

    n_show = min(10, len(photos))
    plt.figure(figsize=(15, 3 * int(np.ceil(n_show / 5))))

    for i, p in enumerate(photos[:n_show], 1):
        img = Image.open(p).convert("RGB")
        x = F.to_tensor(img).to(device)
        out = model([x])[0]
        keep = out["scores"].detach().cpu().numpy() >= score_thr

        img_np = np.array(img)
        pred_union = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=bool)
        if out["masks"].shape[0] > 0 and keep.any():
            pred_masks = out["masks"][keep].detach().cpu().numpy()[:, 0, :, :] >= mask_thr
            pred_union = pred_masks.any(axis=0)

        overlay = img_np.copy()
        overlay[pred_union] = (0.6 * overlay[pred_union] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)

        plt.subplot(int(np.ceil(n_show / 5)), 5, i)
        plt.imshow(overlay)
        plt.title(p.name)
        plt.axis("off")

        if masks_dir is not None and masks_dir.is_dir():
            mask_path = masks_dir / f"{p.stem}.png"
            if mask_path.is_file():
                gt = np.array(Image.open(mask_path).convert("L")) > 0
                iou = binary_iou(pred_union, gt)
                pr, rc = precision_recall(pred_union, gt)
                l2 = l2_distance(pred_union, gt)

                ious.append(iou)
                precisions.append(pr)
                recalls.append(rc)
                l2s.append(l2)
                hits_05 += int(iou >= 0.5)
                hits_075 += int(iou >= 0.75)
                hits_09 += int(iou >= 0.9)
                gt_count += 1

    plt.tight_layout()
    plt.show()

    if gt_count == 0:
        print("Маски не найдены, поэтому тут только визуализация")
        return None

    return {
        "IoU_mean": float(np.mean(ious)),
        "Precision_mean": float(np.mean(precisions)),
        "Recall_mean": float(np.mean(recalls)),
        "L2_mean": float(np.mean(l2s)),
        "IoU>=0.5": hits_05 / gt_count,
        "IoU>=0.75": hits_075 / gt_count,
        "IoU>=0.9": hits_09 / gt_count,
        "N": gt_count,
    }


my_metrics = evaluate_my_photos(model, MY_PHOTOS_DIR, MY_MASKS_DIR, score_thr=0.5, mask_thr=0.5)
if my_metrics is not None:
    print("Метрики на своих фото:")
    for k, v in my_metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")"""
    )
)

cells.append(
    md(
        """## Что писать в отчёте

1. **Какой backbone и модель**  
   Использована `Mask R-CNN ResNet50-FPN` из `torchvision` с предобучением (`MaskRCNN_ResNet50_FPN_Weights.DEFAULT`).  
   Штатные головы классификации и масок заменены под `NUM_CLASSES = 1 + 8`.

2. **Размер train/val, эпохи, batch, устройство**  
   Размеры считаются и печатаются в ноутбуке (`train images`, `val images`).  
   В коде: `EPOCHS = 5`, `BATCH_SIZE = 8` для CUDA (иначе 2), устройство выбирается автоматически (`cuda`/`mps`/`cpu`).

3. **Метрики на val**  
   Выводятся `IoU_mean`, `Precision_mean`, `Recall_mean`, `L2_mean` и доли `IoU>=0.5`, `IoU>=0.75`, `IoU>=0.9`.

4. **Результат на 10 своих фото**  
   Есть визуализация сегментации для первых 10 фото из `my_photos`.  
   Если есть маски в `my_photos_masks`, считаются те же метрики.

5. **Короткий вывод по ошибкам**  
   Типичные проблемы: маленькие знаки, частично закрытые знаки, сильный наклон, дальние объекты и плохой свет.  
   Обычно помогает больше эпох, аугментации и балансировка редких классов.
"""
    )
)

nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}}, nbformat=4, nbformat_minor=5)
out = ROOT / "lab_6.ipynb"
nbformat.write(nb, out)
print("written", out)
