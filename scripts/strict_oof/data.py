"""严格 OOF 的按行加载与无拷贝 RAM Dataset。"""

from __future__ import annotations

import time
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from strict_oof.protocol import (
    CLASS_TO_IDX,
    IMAGE_SIZE,
    load_cache_inventory,
    load_contract,
    paths,
)

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def cache_path(row: dict[str, Any], exp_dir: Path | None = None) -> Path:
    """返回一行对应的 JPEG-384 缓存路径。"""
    root = paths(exp_dir).exp / "big" / "jpeg384"
    name = Path(str(row["path"]).replace("\\", "/")).stem + ".jpg"
    return root / row["cls"] / row["pt"] / name


def load_rows(
    indexed_rows: list[tuple[int, dict[str, Any]]],
    exp_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """只解码当前阶段允许接触的行。

    外折训练调用时不传入 held-out 行，因而该进程不会解码或
    读取 held-out 标签。
    """
    n = len(indexed_rows)
    if n < 1:
        raise ValueError("不允许加载空行集")
    images = np.empty((n, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    labels = np.empty(n, dtype=np.int64)
    global_indices = np.empty(n, dtype=np.int64)
    patients = np.empty(n, dtype=object)
    rows: list[dict[str, Any]] = []
    contract = load_contract(exp_dir, fresh=False)
    inventory = load_cache_inventory(
        exp_dir,
        expected_file_sha256=contract["cache_inventory_file_sha256"],
    )
    started = time.time()
    for local_index, (global_index, row) in enumerate(indexed_rows):
        target = cache_path(row, exp_dir)
        relative = target.relative_to(paths(exp_dir).exp).as_posix()
        frozen = inventory.get(relative)
        if frozen is None or not target.is_file():
            raise FileNotFoundError(f"合同内缓存在运行期缺失/未登记: {relative}")
        raw = target.read_bytes()
        if (
            len(raw) != frozen["size"]
            or hashlib.sha256(raw).hexdigest() != frozen["sha256"]
        ):
            raise RuntimeError(f"合同内 JPEG 字节漂移: {relative}")
        with Image.open(BytesIO(raw)) as image:
            if image.size != (IMAGE_SIZE, IMAGE_SIZE) or image.mode != "RGB":
                raise RuntimeError(
                    f"合同内 JPEG 解码属性漂移: {relative} "
                    f"size={image.size} mode={image.mode}"
                )
            decoded = np.asarray(image)
            if decoded.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                raise RuntimeError(f"合同内 JPEG 解码形状漂移: {relative}")
            images[local_index] = decoded
        labels[local_index] = CLASS_TO_IDX[row["cls"]]
        global_indices[local_index] = global_index
        patients[local_index] = row["pt"]
        rows.append(row)
        if (local_index + 1) % 10_000 == 0:
            print(
                f"load {local_index + 1}/{n} ({time.time() - started:.0f}s)",
                flush=True,
            )
    print(
        f"loaded {n} images ({images.nbytes / 2**30:.1f} GiB, "
        f"{time.time() - started:.0f}s)",
        flush=True,
    )
    return images, labels, global_indices, patients.astype(str), rows


class IndexedRamDataset(Dataset):
    """对共享 RAM 图像数组应用局部索引和确定随机增广。"""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        train: bool,
        seed: int = 0,
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def _augment(self, value: np.ndarray) -> np.ndarray:
        rng = self.rng
        rotations = int(rng.integers(4))
        if rotations:
            value = np.rot90(value, rotations, axes=(0, 1))
        if rng.random() < 0.5:
            value = value[:, ::-1]
        if rng.random() < 0.5:
            value = value[::-1, :]
        if rng.random() < 0.3:
            alpha = 1.0 + rng.uniform(-0.15, 0.15)
            beta = rng.uniform(-12, 12)
            value = np.clip(value.astype(np.float32) * alpha + beta, 0, 255).astype(
                np.uint8
            )
        if rng.random() < 0.7:
            scale = rng.uniform(0.8, 1.0)
            height = width = int(IMAGE_SIZE * scale)
            top = int(rng.integers(0, IMAGE_SIZE - height + 1))
            left = int(rng.integers(0, IMAGE_SIZE - width + 1))
            crop = Image.fromarray(
                np.ascontiguousarray(value[top : top + height, left : left + width])
            )
            value = np.asarray(crop.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))
        return np.ascontiguousarray(value)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        source_index = int(self.indices[index])
        value = self.images[source_index]
        if self.train:
            value = self._augment(value)
        normalized = value.astype(np.float32) / 255.0
        normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1))
        return tensor, int(self.labels[source_index])
