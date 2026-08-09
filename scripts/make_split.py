"""人级（person-level）分层三划分 Train/Meta-Cal/Test = 72/8/20（seed=42）.

档案单位是活检文件夹 `PT编号_日期`；同一 PT 编号 = 同一患者（与项目4数据模型
一致），可能有多次活检。划分以 **患者（PT 前缀）** 为单位分组，患者的全部活
检与图像只进一个子集——比逐文件夹划分更严格，杜绝重复活检跨子集泄漏。

用法: python make_split.py
输出: results/split.json（文件夹→子集）, results/split_stats.json
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict

from config import CLASSES, OUT_DIR, SEED, SPLIT_FRACS


def person_of(folder: str) -> str:
    return folder.split("_")[0]


def main() -> None:
    manifest = json.loads((OUT_DIR / "manifest.json").read_text("utf-8"))

    # 患者 -> {folder: (cls, n_img)}
    person_folders: dict[str, dict[str, str]] = defaultdict(dict)
    person_img: dict[str, Counter] = defaultdict(Counter)
    for row in manifest:
        p = person_of(row["pt"])
        person_folders[p][row["pt"]] = row["cls"]
        person_img[p][row["cls"]] += 1

    # 患者的分层类别 = 其图像数最多的类别（重复活检诊断不一致时取多数）
    person_cls = {p: c.most_common(1)[0][0] for p, c in person_img.items()}
    multi_biopsy = {p: fs for p, fs in person_folders.items() if len(fs) > 1}
    cross_class = {p: sorted(set(fs.values()))
                   for p, fs in person_folders.items()
                   if len(set(fs.values())) > 1}

    # 按类分层、以患者为单位划分
    rng = random.Random(SEED)
    assign_person: dict[str, str] = {}
    for cls in CLASSES:
        persons = sorted(p for p, c in person_cls.items() if c == cls)
        rng.shuffle(persons)
        n = len(persons)
        n_test = max(1, round(n * SPLIT_FRACS["test"]))
        n_mc = max(1, round(n * SPLIT_FRACS["metacal"]))
        for i, p in enumerate(persons):
            assign_person[p] = ("test" if i < n_test
                                else "metacal" if i < n_test + n_mc
                                else "train")

    # 文件夹继承其患者的子集
    assign = {folder: assign_person[person_of(folder)]
              for p, fs in person_folders.items() for folder in fs}

    # ---- 防泄漏自检：人级 + 文件夹级 ----
    subset_person = defaultdict(set)
    for p, s in assign_person.items():
        subset_person[s].add(p)
    overlap = ((subset_person["train"] & subset_person["test"])
               | (subset_person["train"] & subset_person["metacal"])
               | (subset_person["metacal"] & subset_person["test"]))
    assert not overlap, f"患者跨子集: {sorted(overlap)[:5]}"
    for folder, s in assign.items():
        assert assign_person[person_of(folder)] == s
    straddle = [p for p, fs in multi_biopsy.items()
                if len({assign[f] for f in fs}) > 1]
    assert not straddle, f"多活检患者跨子集: {straddle[:5]}"

    # ---- 统计 ----
    stats = {
        "seed": SEED, "fracs": SPLIT_FRACS, "unit": "person (PT prefix)",
        "persons_total": len(person_folders),
        "biopsies_total": sum(len(fs) for fs in person_folders.values()),
        "multi_biopsy_persons": len(multi_biopsy),
        "cross_class_persons": {p: c for p, c in cross_class.items()},
        "patients": {s: len(v) for s, v in subset_person.items()},
        "biopsies": defaultdict(int),
        "images": defaultdict(int),
        "per_class": {c: {s: {"images": 0, "patients": 0, "biopsies": 0}
                          for s in ("train", "metacal", "test")}
                      for c in CLASSES},
    }
    seen_pt_cls = set()
    seen_folder = set()
    for row in manifest:
        s = assign[row["pt"]]
        stats["images"][s] += 1
        stats["per_class"][row["cls"]][s]["images"] += 1
        if row["pt"] not in seen_folder:
            seen_folder.add(row["pt"])
            stats["biopsies"][s] += 1
            stats["per_class"][row["cls"]][s]["biopsies"] += 1
        key = (person_of(row["pt"]), row["cls"], s)
        if key not in seen_pt_cls:
            seen_pt_cls.add(key)
            stats["per_class"][row["cls"]][s]["patients"] += 1

    (OUT_DIR / "split.json").write_text(
        json.dumps(assign, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "split_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"persons={stats['persons_total']} "
          f"biopsies={stats['biopsies_total']} "
          f"multi_biopsy={stats['multi_biopsy_persons']} "
          f"cross_class={len(cross_class)}")
    print("patients:", dict(stats["patients"]))
    print("images:", dict(stats["images"]))
    print("LEAKAGE CHECK PASSED: person-level, no patient crosses subsets")


if __name__ == "__main__":
    main()
