#!/usr/bin/env python3
"""python build_lab5.py -> пишет lab_5.ipynb"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent
cells = []

cells.append(
    new_markdown_cell(
        """# Лабораторная 5 — детекция номеров на фото

В задании речь про номера на улице и длинные строки символов, но стандартный открытый бенчмарк под это почти не подходит целиком, поэтому по части метрик я опираюсь на SVHN, там разметка это отдельные цифры домовых номеров в кадре. Полный номер авто с буквами разных алфавитов потребовал бы своей разметки и отдельного OCR, здесь я делаю рабочий пайплайн детекции цифр как у SVHN и прогоняю его на своих снимках из папки test 1.

Папка test 2 — уже распакованные PNG и digitStruct.mat (без скачивания). На них же и обучаю модель, и считаю IoU и mAP через torchmetrics. Если нужна честная оценка на отдельном сплите, раздели кадры на train и val самостоятельно."""
    )
)

cells.append(
    new_markdown_cell(
        """## Зависимости

На **Windows с видеокартой NVIDIA (например RTX 5070)** сначала поставь **PyTorch с CUDA** с [pytorch.org](https://pytorch.org): выбери Windows, Pip и актуальную ветку CUDA (12.x), выполни показанную команду в терминале или в отдельной ячейке. Без сборки с CUDA `torch.cuda.is_available()` будет False и всё уйдёт на CPU.

Дальше в ячейке ниже ставятся **h5py** (mat v7.3) и **torchmetrics** (mAP)."""
    )
)

cells.append(
    new_code_cell(
        """import subprocess
import sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "h5py", "torchmetrics"]
)"""
    )
)

cells.append(
    new_markdown_cell(
        """## Пути и устройство

Рабочая папка — каталог **lab_5** (рядом лежат **test 1** и **test 2**). Если ноутбук открыт из родительской папки, скрипт сам попробует найти **test 2**.

Устройство: в приоритете **CUDA** (RTX на Windows), затем при необходимости MPS (Mac), иначе CPU."""
    )
)

cells.append(
    new_code_cell(
        """import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

ROOT = Path.cwd()
for cand in (ROOT, ROOT / "lab_5", ROOT.parent / "lab_5"):
    if cand.is_dir() and (cand / "test 2").is_dir():
        ROOT = cand
        break
assert (ROOT / "test 2").is_dir(), (
    "Не найдена папка test 2. Откройте ноутбук из каталога lab_5 или укажите ROOT вручную."
)

TEST1 = ROOT / "test 1"
TEST2 = ROOT / "test 2"
TEST_MAT = TEST2 / "digitStruct.mat"
TRAIN_DIR = TEST2
TRAIN_MAT = TEST_MAT

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("ROOT", ROOT.resolve())
print("train/test PNG", TRAIN_DIR.resolve())
print("device", device, end="")
if device.type == "cuda":
    print(" —", torch.cuda.get_device_name(0))
else:
    print()"""
    )
)

cells.append(
    new_markdown_cell(
        """## Датасет и модель

Импортирую готовый разбор bbox из svhn_bbox.py, собираю Faster R-CNN с претрейном на COCO и головой на 11 классов, фон плюс 10 цифр как в разметке SVHN."""
    )
)

cells.append(
    new_code_cell(
        """sys.path.insert(0, str(ROOT))
from svhn_bbox import SVHNDetectionDataset, collate_fn, get_model

MAX_TRAIN_IMAGES = None
BATCH = 8


def to_tensor_eval(img, target):
    return F.to_tensor(img), target


class TDataset(SVHNDetectionDataset):
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        return to_tensor_eval(img, target)


train_ds = TDataset(TRAIN_DIR, TRAIN_MAT, max_images=MAX_TRAIN_IMAGES)
train_loader = DataLoader(
    train_ds, batch_size=BATCH, shuffle=True, num_workers=0, collate_fn=collate_fn
)

model = get_model(num_classes=11).to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=1e-4)
lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
print("train размер", len(train_ds))"""
    )
)

cells.append(
    new_markdown_cell(
        """## Обучение

Несколько эпох SGD. На **RTX с CUDA** обычно комфортно; на CPU будет долго. При нехватке VRAM уменьши **BATCH** в предыдущей ячейке. При необходимости подними **EPOCHS** или ограничь выборку через **MAX_TRAIN_IMAGES** (сейчас None — все png из test 2)."""
    )
)

cells.append(
    new_code_cell(
        """EPOCHS = 4


def train_one_epoch():
    model.train()
    losses_acc = []
    for images, targets in train_loader:
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        losses_acc.append(losses.item())
    return float(np.mean(losses_acc))


for ep in range(1, EPOCHS + 1):
    m = train_one_epoch()
    lr_sched.step()
    print(f"epoch {ep}/{EPOCHS} mean loss {m:.4f}")

torch.save(model.state_dict(), ROOT / "frcnn_svhn.pt")
print("веса сохранены", ROOT / "frcnn_svhn.pt")
train_ds.close()"""
    )
)

cells.append(
    new_markdown_cell(
        """## Метрики на test 2

Собираю предсказания и цели в формате torchmetrics, классы привожу к 0..9 через остаток по 10 чтобы и эталон SVHN и предсказания были в одном пространстве. Считаю mAP, отдельно можно смотреть вывод словаря метрики."""
    )
)

cells.append(
    new_code_cell(
        """def svhn_to_cls10(t):
    return (t % 10).long()


@torch.no_grad()
def evaluate_folder(img_root: Path, mat_path: Path, max_images=None):
    ds = SVHNDetectionDataset(img_root, mat_path, max_images=max_images)
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    for i in range(len(ds)):
        pil, tgt = ds[i]
        img = F.to_tensor(pil).to(device)
        out = model([img])[0]
        keep = out["scores"] >= 0.5
        pb = out["boxes"][keep].cpu()
        pl = svhn_to_cls10(out["labels"][keep].cpu())
        ps = out["scores"][keep].cpu()
        tb = tgt["boxes"].cpu()
        tl = svhn_to_cls10(tgt["labels"]).cpu()
        metric.update(
            [dict(boxes=pb, scores=ps, labels=pl)],
            [dict(boxes=tb, labels=tl)],
        )
    ds.close()
    return metric.compute()


model.load_state_dict(torch.load(ROOT / "frcnn_svhn.pt", map_location=device))
model.eval()
res = evaluate_folder(TEST2, TEST_MAT, max_images=None)
print(res)
map50 = float(res.get("map_50", res.get("map", 0.0)))
print("mAP при IoU 0.5 около", map50)
for k in ("map", "map_50", "map_75", "mar_100", "mar_10"):
    if k in res:
        print(k, float(res[k]))"""
    )
)

cells.append(
    new_markdown_cell(
        """## Свои фото test 1

Здесь только визуализация боксов без численной метрики, эталона разметки в папке нет."""
    )
)

cells.append(
    new_code_cell(
        """from PIL import Image

model.load_state_dict(torch.load(ROOT / "frcnn_svhn.pt", map_location=device))
model.to(device)
model.eval()

paths = sorted(TEST1.glob("*.png"))
if not paths:
    print("нет png в test 1")
else:
    n = min(3, len(paths))
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, p in zip(axes, paths[:3]):
        im = F.to_tensor(Image.open(p).convert("RGB")).to(device)
        with torch.no_grad():
            out = model([im])[0]
        img_np = im.cpu().permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img_np, 0, 1))
        for b, s, lab in zip(out["boxes"], out["scores"], out["labels"]):
            if s < 0.5:
                continue
            x1, y1, x2, y2 = b.cpu().numpy()
            ax.add_patch(
                plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=1.5)
            )
        ax.set_title(p.name)
        ax.axis("off")
    plt.tight_layout()
    plt.show()"""
    )
)

cells.append(
    new_markdown_cell(
        """## Что сказать в отчёте

Если mAP на test 2 ниже 0.6, увеличь EPOCHS или попробуй отдельный train-сплит (сейчас обучение на тех же кадрах, что и метрики — цифры будут завышены относительно новых фото). Для настоящих госномеров с буквами нужен другой датасет и как минимум отдельный распознаватель символов после детекции."""
    )
)

nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}}, nbformat=4, nbformat_minor=5)
out = ROOT / "lab_5.ipynb"
nbformat.write(nb, out)
print("written", out)
