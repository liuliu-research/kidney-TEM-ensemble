"""将 3 个缺失单独消融臂与旧 6 臂合并为全扫描。"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import f1_score

from strict_oof.protocol import CLASSES
from strict_oof.analyze import calibration_metrics, full_metrics, tta_mean
from strict_oof.predict import prediction_paths
from strict_oof.protocol import (
    atomic_write_json,
    file_sha256,
    load_contract,
    paths,
)

ARMS = ("cb_only", "mixup_only", "cutmix_only")


def input_prediction_sha256(contract: dict[str, object]) -> dict[str, str]:
    """绑定六个新消融预测 bundle，防止上游重做后复用旧分析。"""
    root = paths().exp
    result: dict[str, str] = {}
    for arm in ARMS:
        for subset in ("metacal", "test"):
            bundle, _shards, _meta, _lock = prediction_paths(
                "ablation",
                str(contract["additional_ablation_backbone"]),
                subset,
                arm=arm,
            )
            result[bundle.resolve().relative_to(root).as_posix()] = file_sha256(bundle)
    return result


def main() -> int:
    contract = load_contract()
    from strict_oof.preflight import assert_frozen_runtime

    assert_frozen_runtime(contract)
    p = paths()
    output = p.out / "extended_ablation_analysis.json"
    input_hashes = input_prediction_sha256(contract)
    if output.is_file():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "complete"
                and existing.get("contract_sha256") == contract["contract_sha256"]
                and existing.get("input_prediction_sha256") == input_hashes
            ):
                print("extended ablation analysis complete, skip")
                return 0
        except (OSError, json.JSONDecodeError):
            pass
    legacy_path = p.exp / "results" / "ablation_analysis.json"
    if file_sha256(legacy_path) != contract["legacy_ablation_analysis_sha256"]:
        raise RuntimeError("旧消融分析与合同不匹配")
    legacy_analysis = json.loads(legacy_path.read_text(encoding="utf-8"))
    combined: dict[str, object] = {}
    for key, value in legacy_analysis.items():
        if key.startswith("_"):
            continue
        row = dict(value)
        flags = dict(row["flags"])
        flags.setdefault("mix", bool(flags.get("mix", False)))
        flags.setdefault("mix_mode", "alternating" if flags["mix"] else "none")
        row["flags"] = flags
        row["backbone"] = contract["additional_ablation_backbone"]
        row["evaluation_tta"] = 1
        row.setdefault(
            "calibration",
            {"top_label_ece": row.get("ece"), "nll": row.get("nll")},
        )
        row.setdefault(
            "brier_multiclass",
            row.get("calibration", {}).get("brier_multiclass"),
        )
        row["contrast_interpretation"] = (
            "descriptive single-run arm; no paired or across-seed "
            "uncertainty for arm differences"
        )
        combined[key] = row
    new_arms: dict[str, object] = {}
    for arm in ARMS:
        bundles = {}
        for subset in ("metacal", "test"):
            path, _shards, _meta, _lock = prediction_paths(
                "ablation",
                contract["additional_ablation_backbone"],
                subset,
                arm=arm,
            )
            with np.load(path, allow_pickle=False) as archive:
                if (
                    str(archive["contract_sha256"].item())
                    != contract["contract_sha256"]
                ):
                    raise RuntimeError(f"{arm}/{subset} 合同不匹配")
                bundles[subset] = {
                    "probs": archive["probs"].copy(),
                    "labels": archive["labels"].copy(),
                }
        # 与旧六臂 ablation_analysis.py 严格统一为 TTA1，否则九臂不可比。
        metacal_prob = tta_mean(bundles["metacal"]["probs"], 1)
        test_prob = tta_mean(bundles["test"]["probs"], 1)
        y_metacal = bundles["metacal"]["labels"]
        y_test = bundles["test"]["labels"]
        calibration = calibration_metrics(y_test, test_prob)
        flags = dict(contract["additional_ablation_arms"][arm])
        flags["mix"] = flags["mix_mode"] != "none"
        predictions = test_prob.argmax(1)
        pred_share = np.bincount(predictions, minlength=len(CLASSES)) / len(y_test)
        true_share = np.bincount(y_test, minlength=len(CLASSES)) / len(y_test)
        row = {
            "flags": flags,
            "backbone": contract["additional_ablation_backbone"],
            "evaluation_tta": 1,
            "metacal_macro_f1": float(
                f1_score(
                    y_metacal,
                    metacal_prob.argmax(1),
                    average="macro",
                    zero_division=0,
                )
            ),
            "metrics": full_metrics(y_test, test_prob),
            "calibration": calibration,
            "ece": calibration["top_label_ece"],
            "nll": calibration["nll"],
            "brier_multiclass": calibration["brier_multiclass"],
            "pred_vs_true_share": {
                class_name: {
                    "pred": float(pred_share[class_index]),
                    "true": float(true_share[class_index]),
                }
                for class_index, class_name in enumerate(CLASSES)
            },
            "minority_overcompensation_index": float(
                (pred_share - true_share)[true_share < 0.05].sum()
            ),
            "test_labels_used_for_training_or_epoch_selection": False,
            "contrast_interpretation": (
                "descriptive single-run arm; no paired or across-seed "
                "uncertainty for arm differences"
            ),
        }
        new_arms[arm] = row
        combined[arm] = row
    result = {
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "input_prediction_sha256": input_hashes,
        "backbone": contract["additional_ablation_backbone"],
        "legacy_source_sha256": contract["legacy_ablation_analysis_sha256"],
        "legacy_arms": sorted(
            key for key in legacy_analysis if not key.startswith("_")
        ),
        "new_arms": list(ARMS),
        "arms": combined,
        "new_arm_details": new_arms,
        "inference_boundary": (
            "All between-arm differences are descriptive fixed-budget, "
            "single-run contrasts. They do not establish statistical "
            "significance or a causal component effect."
        ),
    }
    atomic_write_json(output, result)
    print(json.dumps(result["new_arm_details"], indent=2)[:1600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
