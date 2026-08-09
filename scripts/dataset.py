"""RAM 缓存数据集：JPEG-384 解码为 uint8 常驻内存（数据治理层）.

约 60k × 384×384×3 uint8 ≈ 27 GB，GB10 统一内存 121 GB 可容纳。
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import CACHE_DIR, CLASS_TO_IDX, OUT_DIR

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_subset(subset: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """返回 (images uint8 NHWC, labels int64, rows)。"""
    manifest = json.loads((OUT_DIR / "manifest.json").read_text("utf-8"))
    split = json.loads((OUT_DIR / "split.json").read_text("utf-8"))
    rows, dropped = [], 0
    for r in manifest:
        if split[r["pt"]] != subset:
            continue
        p = CACHE_DIR / r["cls"] / r["pt"] / (
            r["path"].rsplit("/", 1)[-1].rsplit(".", 1)[0] + ".jpg")
        if p.exists():
            rows.append(r)
        else:
            dropped += 1          # 损坏源文件（QC 排除，见 cache_report.json）
    if dropped:
        print(f"  {subset}: {dropped} corrupted-source images excluded (QC)",
              flush=True)
    n = len(rows)
    imgs = np.empty((n, 384, 384, 3), dtype=np.uint8)
    labels = np.empty(n, dtype=np.int64)
    t0 = time.time()
    for i, r in enumerate(rows):
        p = CACHE_DIR / r["cls"] / r["pt"] / (r["path"].rsplit("/", 1)[-1]
                                              .rsplit(".", 1)[0] + ".jpg")
        with Image.open(p) as im:
            imgs[i] = np.asarray(im.convert("RGB"))
        labels[i] = CLASS_TO_IDX[r["cls"]]
        if (i + 1) % 10000 == 0:
            print(f"  load {subset} {i+1}/{n} ({time.time()-t0:.0f}s)",
                  flush=True)
    print(f"  {subset}: {n} images in RAM ({imgs.nbytes/2**30:.1f} GiB, "
          f"{time.time()-t0:.0f}s)", flush=True)
    return imgs, labels, rows


class RamDataset(Dataset):
    """训练/评估共用；train=True 时做几何+光度增广（uint8 域，后归一化）。"""

    def __init__(self, imgs: np.ndarray, labels: np.ndarray,
                 train: bool, seed: int = 0):
        self.imgs, self.labels, self.train = imgs, labels, train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def _augment(self, a: np.ndarray) -> np.ndarray:
        rng = self.rng
        k = int(rng.integers(4))
        if k:
            a = np.rot90(a, k, axes=(0, 1))
        if rng.random() < 0.5:
            a = a[:, ::-1]
        if rng.random() < 0.5:
            a = a[::-1, :]
        if rng.random() < 0.3:                      # 亮度/对比度抖动
            alpha = 1.0 + rng.uniform(-0.15, 0.15)
            beta = rng.uniform(-12, 12)
            a = np.clip(a.astype(np.float32) * alpha + beta, 0, 255
                        ).astype(np.uint8)
        if rng.random() < 0.7:                      # 随机缩放裁剪 0.8–1.0
            s = rng.uniform(0.8, 1.0)
            h = w = int(384 * s)
            y = int(rng.integers(0, 384 - h + 1))
            x = int(rng.integers(0, 384 - w + 1))
            crop = Image.fromarray(np.ascontiguousarray(a[y:y+h, x:x+w]))
            a = np.asarray(crop.resize((384, 384), Image.BILINEAR))
        return np.ascontiguousarray(a)

    def __getitem__(self, i: int):
        a = self.imgs[i]
        if self.train:
            a = self._augment(a)
        x = a.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(x.transpose(2, 0, 1)), int(self.labels[i])


def class_counts(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.bincount(labels, minlength=num_classes)
