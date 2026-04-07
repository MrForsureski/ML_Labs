"""
Разбор digitStruct.mat и подготовка целей для Faster R-CNN
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _read_name(f: h5py.File, index: int) -> str:
    ref = f["digitStruct"]["name"][index, 0]
    arr = f[ref][:]
    return "".join(chr(int(x[0])) for x in arr)


def _read_field_list(f: h5py.File, bb, key: str) -> List[float]:
    arr = bb[key][()]
    if arr.dtype == object:
        vals = []
        for i in range(arr.shape[0]):
            ref = arr[i, 0]
            vals.append(float(np.array(f[ref][()]).flat[0]))
        return vals
    return [float(np.array(arr).flat[0])]


def parse_image_entry(f: h5py.File, index: int) -> Tuple[np.ndarray, np.ndarray]:
    """Возвращает boxes (N,4) xyxy в пикселях и labels (N,) со значениями 1-10 как в SVHN"""
    bb = f[f["digitStruct"]["bbox"][index, 0]]
    left = _read_field_list(f, bb, "left")
    top = _read_field_list(f, bb, "top")
    width = _read_field_list(f, bb, "width")
    height = _read_field_list(f, bb, "height")
    label = _read_field_list(f, bb, "label")
    n = len(left)
    boxes = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        l, t, w, h = left[i], top[i], width[i], height[i]
        boxes[i] = [l, t, l + w, t + h]
    labels = np.array(label, dtype=np.int64)
    return boxes, labels


class SVHNDetectionDataset(Dataset):
    def __init__(self, root: str | Path, mat_path: str | Path, max_images: int | None = None):
        super().__init__()
        self.root = Path(root)
        self.mat_path = Path(mat_path)
        self._h5: h5py.File | None = None
        ids = sorted(int(p.stem) for p in self.root.glob("*.png") if p.stem.isdigit())
        if max_images is not None:
            ids = ids[:max_images]
        self.ids = ids

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_h5"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._h5 = None

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.mat_path, "r")
        return self._h5

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        file_id = self.ids[idx]
        mat_idx = file_id - 1
        path = self.root / f"{file_id}.png"
        img = Image.open(path).convert("RGB")
        w, h = img.size
        boxes, labels = parse_image_entry(self._file(), mat_idx)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes[:, 0::2].clamp_(0, w)
        boxes[:, 1::2].clamp_(0, h)
        target = {"boxes": boxes, "labels": labels}
        return img, target

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes: int = 11, backbone: str = "mobilenet_v3"):
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    if  backbone in ("mobilenet_v3", "mobilenet"):
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
        model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")


    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
