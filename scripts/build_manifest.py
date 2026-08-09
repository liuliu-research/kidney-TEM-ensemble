"""扫描物理文件夹 → 数据清单 manifest.json + 数据集统计表（论文 Table 1 数据源）.

用法: python build_manifest.py
输出: results/manifest.json, results/dataset_stats.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime

from config import CLASS_FOLDERS, DATA_ROOT, IMG_EXTS, OUT_DIR


def main() -> None:
    manifest = []          # [{class, biopsy-folder, path}]
    stats = defaultdict(lambda: {"biopsies": 0, "images": 0,
                                 "img_per_biopsy": [], "persons": set()})
    bad = []

    for code, folder in CLASS_FOLDERS.items():
        img_root = DATA_ROOT / folder / "电镜图片"
        if not img_root.is_dir():
            raise FileNotFoundError(img_root)
        for pt_dir in sorted(p for p in img_root.iterdir() if p.is_dir()):
            imgs = sorted(
                f for f in pt_dir.iterdir()
                if f.is_file() and f.suffix in IMG_EXTS
            )
            if not imgs:
                bad.append(str(pt_dir))
                continue
            stats[code]["biopsies"] += 1
            stats[code]["persons"].add(pt_dir.name.split("_")[0])
            stats[code]["images"] += len(imgs)
            stats[code]["img_per_biopsy"].append(len(imgs))
            for f in imgs:
                manifest.append(
                    {"cls": code, "pt": pt_dir.name, "path": str(f)}
                )

    all_persons = set()
    for v in stats.values():
        all_persons |= v["persons"]
    total_bio = sum(v["biopsies"] for v in stats.values())
    total_img = sum(v["images"] for v in stats.values())
    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "total_patients": len(all_persons),      # 唯一 PT 前缀（人）
        "total_biopsies": total_bio,             # 活检文件夹
        "total_images": total_img,
        "empty_patient_dirs": bad,
        "per_class": {
            c: {
                "patients": len(v["persons"]),
                "biopsies": v["biopsies"],
                "images": v["images"],
                "images_per_patient_mean": round(
                    v["images"] / len(v["persons"]), 2
                ),
                "images_per_biopsy_min": min(v["img_per_biopsy"]),
                "images_per_biopsy_max": max(v["img_per_biopsy"]),
            }
            for c, v in stats.items()
        },
    }
    total_pt = len(all_persons)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "dataset_stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["per_class"], ensure_ascii=False, indent=1))
    print(f"TOTAL patients={total_pt} images={total_img} empty_dirs={len(bad)}")


if __name__ == "__main__":
    main()
