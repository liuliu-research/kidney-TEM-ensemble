"""消融批次分析（回应 R3.4 / R4.6 / R5.2 / R5.10）.

对 6 个消融臂（Swin-B, 30-epoch 预算）：
  · Test 全指标 + 少数类逐类 recall
  · 过度补偿探针：预测分布偏移(每类预测占比 vs 真实占比)、ECE、NLL
输出: results/ablation_analysis.json
"""
from __future__ import annotations

import json

import numpy as np

import config as C
from fusion_analysis import calib_suite, full_metrics, load_preds, tta_mean


def main() -> None:
    out = {}
    for abl in C.ABLATIONS:
        tag = f"swin_base_abl_{abl}"
        try:
            probs, y, pts, _ = load_preds(tag, "test")
        except FileNotFoundError:
            print("WARN missing", tag)
            continue
        p = tta_mean(probs, 1)
        m = full_metrics(y, p)
        cal = calib_suite(y, p)
        pred_share = np.bincount(p.argmax(1),
                                 minlength=C.NUM_CLASSES) / len(y)
        true_share = np.bincount(y, minlength=C.NUM_CLASSES) / len(y)
        out[abl] = {
            "flags": C.ABLATIONS[abl],
            "metrics": m,
            "ece": cal["ece"],
            "nll": cal["nll"],
            "brier_multiclass": cal["brier_multiclass"],
            "pred_vs_true_share": {
                C.CLASSES[c]: {"pred": float(pred_share[c]),
                               "true": float(true_share[c])}
                for c in range(C.NUM_CLASSES)},
            "minority_overcompensation_index": float(
                (pred_share - true_share)[true_share < 0.05].sum()),
        }
        print(abl, "macroF1=", round(m["macro_f1"], 4),
              "ECE=", round(cal["ece"], 4), flush=True)

    (C.OUT_DIR / "ablation_analysis.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
