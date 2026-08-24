"""TIFF → JPEG-384 缓存构建与完整性校验.

真实缓存链为：源图 → RGB → 全图等向 384×384 Lanczos 重采样 →
底部 28 行置黑 → JPEG q92。重采样时所有源像素都参与，但最终网络输入会
有意遮蔽 28/384（7.29%）的底部区域以降低页脚元数据捷径风险。该措施不能
证明遮蔽条之外没有文字或测量标记。多进程（20 核）。断点续跑：已存在
且可读则跳过。

用法: python build_cache.py [--workers 18]
输出: big/jpeg384/<cls>/<pt>/<name>.jpg, results/cache_report.json
"""
from __future__ import annotations

import argparse
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image

from config import CACHE_DIR, IMG_SIZE, JPEG_QUALITY, OUT_DIR

ANNOT_STRIP_PX = 28  # 底部遮蔽高度（384 的 7.29%）

Image.MAX_IMAGE_PIXELS = None


def resize_full_frame(image: Image.Image) -> Image.Image:
    """执行缓存链的 RGB 全图重采样步骤。"""
    return image.convert("RGB").resize(
        (IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS
    )


def mask_footer_strip(resized: Image.Image) -> Image.Image:
    """将重采样图像底部固定行数置黑。"""
    if resized.size != (IMG_SIZE, IMG_SIZE):
        raise ValueError(
            f"页脚遮蔽要求 {IMG_SIZE}×{IMG_SIZE}，实际为 {resized.size}"
        )
    array = np.asarray(resized.convert("RGB")).copy()
    array[-ANNOT_STRIP_PX:, :, :] = 0
    return Image.fromarray(array)


def prepare_cache_image(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """返回（仅重采样，重采样+页脚遮蔽）两个阶段。"""
    resized = resize_full_frame(image)
    return resized, mask_footer_strip(resized)


def save_cache_jpeg(
    image: Image.Image, destination: str | Path | BinaryIO
) -> None:
    """按生产缓存完全相同的 Pillow 参数写入 JPEG。"""
    image.save(destination, "JPEG", quality=JPEG_QUALITY)


def encode_cache_jpeg(image: Image.Image) -> bytes:
    """按生产缓存参数返回 JPEG 字节。"""
    buffer = io.BytesIO()
    save_cache_jpeg(image, buffer)
    return buffer.getvalue()


def convert_one(args: tuple[str, str]) -> tuple[str, str]:
    src, dst = args
    dst_p = Path(dst)
    if dst_p.exists() and dst_p.stat().st_size > 0:
        return src, "skip"
    try:
        with Image.open(src) as im:
            _, im = prepare_cache_image(im)
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            save_cache_jpeg(im, dst)
        return src, "ok"
    except Exception as e:  # noqa: BLE001
        return src, f"ERR:{e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=18)
    args = ap.parse_args()

    manifest = json.loads((OUT_DIR / "manifest.json").read_text("utf-8"))
    jobs = []
    for row in manifest:
        src = Path(row["path"])
        dst = CACHE_DIR / row["cls"] / row["pt"] / (src.stem + ".jpg")
        jobs.append((str(src), str(dst)))

    t0 = time.time()
    ok = skip = 0
    errs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (src, status) in enumerate(ex.map(convert_one, jobs, chunksize=64)):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                errs.append({"src": src, "err": status})
            if (i + 1) % 5000 == 0:
                print(f"{i+1}/{len(jobs)} ok={ok} skip={skip} err={len(errs)} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    report = {
        "total": len(jobs), "converted": ok, "skipped": skip,
        "errors": errs, "elapsed_s": round(time.time() - t0, 1),
        "img_size": IMG_SIZE, "jpeg_quality": JPEG_QUALITY,
        "annotation_strip_masked_px": ANNOT_STRIP_PX,
        "annotation_strip_masked_fraction": ANNOT_STRIP_PX / IMG_SIZE,
        "preprocessing_schema": "rgb-resize-mask-jpeg-v2",
    }
    (OUT_DIR / "cache_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "errors"},
                     indent=1), f"errors={len(errs)}")


if __name__ == "__main__":
    main()
