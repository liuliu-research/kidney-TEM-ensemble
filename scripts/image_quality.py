"""TIFF ↔ JPEG-384 图像质量定量（回应 R2.2 / R5.8）.

两层分解：
  A) 降采样损失: TIFF 原图(灰度化) vs 无损 384 (LANCZOS) 上采回原尺寸 → SSIM/PSNR
  B) JPEG 压缩附加损失: 无损 384 vs JPEG-q92 384 → SSIM/PSNR/LPIPS(若可用)
每类分层随机抽样 --per-class 张（默认 30，共 330 张），报告均值±SD 与逐类。

用法: python image_quality.py [--per-class 30]
输出: results/image_quality.json
"""
from __future__ import annotations

import argparse
import io
import json
import random

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from config import CLASSES, IMG_SIZE, JPEG_QUALITY, OUT_DIR, SEED

Image.MAX_IMAGE_PIXELS = None

try:
    import lpips as lpips_lib
    import torch

    _LPIPS = lpips_lib.LPIPS(net="alex")
    _LPIPS.eval()
except Exception:  # noqa: BLE001
    _LPIPS = None


def lpips_dist(a: np.ndarray, b: np.ndarray) -> float | None:
    """a,b: HxW uint8 灰度."""
    if _LPIPS is None:
        return None
    import torch

    def to_t(x: np.ndarray) -> "torch.Tensor":
        t = torch.from_numpy(x).float() / 127.5 - 1.0
        return t[None, None].repeat(1, 3, 1, 1)

    with torch.no_grad():
        return float(_LPIPS(to_t(a), to_t(b)).item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=30)
    args = ap.parse_args()

    manifest = json.loads((OUT_DIR / "manifest.json").read_text("utf-8"))
    rng = random.Random(SEED)
    by_cls: dict[str, list[dict]] = {c: [] for c in CLASSES}
    for row in manifest:
        by_cls[row["cls"]].append(row)

    per_cls_out = {}
    pooled = {k: [] for k in ("ssim_down", "psnr_down", "ssim_jpeg",
                              "psnr_jpeg", "lpips_jpeg")}
    for cls in CLASSES:
        sample = rng.sample(by_cls[cls], min(args.per_class, len(by_cls[cls])))
        vals = {k: [] for k in pooled}
        for row in sample:
            with Image.open(row["path"]) as im:
                gray = np.asarray(im.convert("L"))
                small = im.convert("L").resize(
                    (IMG_SIZE, IMG_SIZE), Image.LANCZOS
                )
            # A) 降采样损失（384 上采回原尺寸对比原图）
            up = np.asarray(
                small.resize(gray.shape[::-1], Image.LANCZOS)
            )
            vals["ssim_down"].append(ssim(gray, up, data_range=255))
            vals["psnr_down"].append(psnr(gray, up, data_range=255))
            # B) JPEG 压缩附加损失（同尺寸 384 对比）
            buf = io.BytesIO()
            small.save(buf, "JPEG", quality=JPEG_QUALITY)
            buf.seek(0)
            jp = np.asarray(Image.open(buf).convert("L"))
            sm = np.asarray(small)
            vals["ssim_jpeg"].append(ssim(sm, jp, data_range=255))
            vals["psnr_jpeg"].append(psnr(sm, jp, data_range=255))
            d = lpips_dist(sm, jp)
            if d is not None:
                vals["lpips_jpeg"].append(d)
        per_cls_out[cls] = {
            k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n": len(v)}
            for k, v in vals.items() if v
        }
        for k, v in vals.items():
            pooled[k].extend(v)

    out = {
        "protocol": {
            "per_class_sample": args.per_class,
            "downsample": f"L LANCZOS -> {IMG_SIZE}",
            "jpeg_quality": JPEG_QUALITY,
            "lpips": "alex" if _LPIPS is not None else "unavailable",
        },
        "overall": {
            k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n": len(v)}
            for k, v in pooled.items() if v
        },
        "per_class": per_cls_out,
    }
    (OUT_DIR / "image_quality.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps(out["overall"], indent=1))


if __name__ == "__main__":
    main()
