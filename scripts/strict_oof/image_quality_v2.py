"""在严格隔离目录生成正式 image-quality v2 报告。"""

from __future__ import annotations

import argparse
import json

import config as legacy
from image_quality import AlexNetLPIPS, analyse
from strict_oof.protocol import atomic_write_json, paths, validate_image_quality_v2


def main() -> int:
    """运行每类 30 张的真实缓存链质量检查。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lpips-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--lpips-max-side", type=int, default=512)
    args = parser.parse_args()
    p = paths()
    target = p.out / "image_quality.json"
    if target.is_file():
        try:
            validate_image_quality_v2(p.exp)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            pass
        else:
            print("strict image-quality v2 complete, skip")
            return 0
    manifest = json.loads(
        (p.exp / "results" / "manifest.json").read_text(encoding="utf-8")
    )
    perceptual = AlexNetLPIPS(args.lpips_device, args.lpips_max_side)
    result = analyse(
        manifest=manifest,
        per_class=30,
        seed=int(legacy.SEED),
        cache_dir=legacy.CACHE_DIR,
        perceptual=perceptual,
    )
    atomic_write_json(target, result)
    validate_image_quality_v2(p.exp)
    print(
        f"strict image-quality v2 complete: "
        f"{result['sampling']['succeeded']}/{result['sampling']['attempted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
