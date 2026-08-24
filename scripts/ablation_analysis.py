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
    # 骨干名从 orchestrator 落盘的选择结果读取，不硬编码——它是步骤名与
    # 产物名之间唯一的桥，写死等于埋一颗改配置就爆的雷。
    bb = json.loads(
        (C.OUT_DIR / "ablation_backbone.json").read_text("utf-8"))["backbone"]
    out, missing = {}, []
    for abl in C.ABLATIONS:
        tag = f"{bb}_abl_{abl}"
        try:
            probs, y, pts, _ = load_preds(tag, "test")
        except Exception as e:  # noqa: BLE001  含 FileNotFoundError/BadZipFile
            # 原实现只捕 FileNotFoundError、打一行 WARN 后 continue，最后
            # 无条件写盘并返回 0 → 任何一个消融臂缺失或 npz 截断，都会产出
            # **残缺的消融表而流水线报告成功**，一路写到 ALL.done，直到出稿
            # 才可能发现。消融表是 R3.4/R4.6/R5.2/R5.10 四条意见的直接证据，
            # 残表进正文即为不实陈述，故改为记录后在末尾显式失败。
            print("MISSING", tag, repr(e), flush=True)
            missing.append(tag)
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

    out["_backbone"] = bb        # 六个消融臂名均不以 _ 开头，不会撞键
    (C.OUT_DIR / "ablation_analysis.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    if missing:
        # 先写盘再失败：残表保留供人工判断，但绝不标记该步成功
        raise SystemExit(
            f"消融臂缺失/损坏 {len(missing)}/{len(C.ABLATIONS)}: {missing}"
            " —— 拒绝以不完整的消融表标记成功")


if __name__ == "__main__":
    main()
