"""TIFF → JPEG-384 全幅缓存构建（不裁剪，等比信息保留）+ 完整性校验.

原图 2048×2063 调色板(P)模式 → RGB → 全幅 resize 到 384×384（纵横比畸变 ≤0.8%，
无任何区域被丢弃）→ JPEG q92。多进程（20 核）。断点续跑：已存在且可读则跳过。

用法: python build_cache.py [--workers 18]
输出: big/jpeg384/<cls>/<pt>/<name>.jpg, results/cache_report.json
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from config import CACHE_DIR, IMG_SIZE, JPEG_QUALITY, OUT_DIR

ANNOT_STRIP_PX = 28   # 底部注释遮蔽高度（384 的 ~7.3%）

Image.MAX_IMAGE_PIXELS = None


def convert_one(args: tuple[str, str]) -> tuple[str, str]:
    src, dst = args
    dst_p = Path(dst)
    if dst_p.exists() and dst_p.stat().st_size > 0:
        return src, "skip"
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            im = im.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            # 遮蔽底部注释条带（烧录的标尺/放大倍数文字，防捷径学习）：
            # 全库注释均位于底部 ~7%（抽样核验 2026-08-08），统一涂黑 28px。
            import numpy as np
            a = np.asarray(im).copy()
            a[-ANNOT_STRIP_PX:, :, :] = 0
            im = Image.fromarray(a)
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=JPEG_QUALITY)
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
    }
    (OUT_DIR / "cache_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "errors"},
                     indent=1), f"errors={len(errs)}")


if __name__ == "__main__":
    main()
