"""严格 OOF 合同、单步与全产物失败关闭验收。"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from strict_oof.predict import _load_prediction_rows, prediction_paths
from strict_oof.protocol import (
    DELIVERY_LEGACY_FILES,
    FINAL_TTA_COUNT,
    NUM_CLASSES,
    OOF_TTA_COUNT,
    OUTER_FOLDS,
    TRAIN_MODELS,
    atomic_write_json,
    file_sha256,
    load_contract,
    paths,
    patient_of,
    source_rows,
    terminal_epoch,
    validate_image_quality_v2,
)
from strict_oof.train import artifact_paths
from strict_oof.workflow import (
    EXPECTED_TOTAL_STEPS,
    EXPECTED_TRAINING_RUNS,
    build_plan,
)

EXPECTED_TEST_BIOPSIES = 1_657
EXPECTED_TEST_PATIENTS = 1_649
EXPECTED_ELIGIBLE_PATIENTS = 1_642
EXPECTED_MIXED_PATIENTS = 7
BOOTSTRAP_EXPECTED = 1_000


def _module_command(args: tuple[str, ...]) -> list[str]:
    first, *rest = args
    if first.startswith("strict_oof/") and first.endswith(".py"):
        return [
            sys.executable,
            "-m",
            first.removesuffix(".py").replace("/", "."),
            *rest,
        ]
    return [sys.executable, *args]


def _marker_output_paths(name: str) -> list[Path]:
    """独立重建当步应绑定的唯一产物集合。"""
    p = paths()
    if name == "image_quality_v2":
        return [p.out / "image_quality.json"]
    if name == "prepare_contract":
        return [p.contract]
    parts = _train_step_parts(name)
    if parts:
        stage, model, fold, arm = parts
        if stage == "ablation":
            model = load_contract(fresh=False)["additional_ablation_backbone"]
        checkpoint, _resume, log, _lock = artifact_paths(
            stage, model, fold=fold, arm=arm
        )
        return [checkpoint, log]
    pred = _prediction_step_parts(name)
    if pred:
        stage, model, subset, fold, arm = pred
        if stage == "ablation":
            model = load_contract(fresh=False)["additional_ablation_backbone"]
        bundle, _shards, meta, _lock = prediction_paths(
            stage, model, subset, fold=fold, arm=arm
        )
        return [bundle, meta]
    return {
        "strict_analysis": [p.out / "analysis.json"],
        "strict_gradcam": [p.out / "gradcam_index.json", p.out / "gradcam"],
        "extended_ablation_analysis": [p.out / "extended_ablation_analysis.json"],
        "validate_strict_oof": [p.validation_report],
        "finalize_strict_oof": [
            p.ready,
            p.delivery,
            p.exp / "results" / "strict_oof_ACTIVE.json",
        ],
    }.get(name, [])


def _current_artifact_hashes(name: str) -> dict[str, str]:
    p = paths()
    hashes: dict[str, str] = {}
    for target in _marker_output_paths(name):
        current_files = (
            sorted(target.rglob("*"), key=lambda item: item.as_posix())
            if target.is_dir()
            else [target]
        )
        for current in current_files:
            if current.is_file():
                hashes[current.resolve().relative_to(p.exp.resolve()).as_posix()] = (
                    file_sha256(current)
                )
    return hashes


def validate_marker(name: str, *, validate_artifact: bool = True) -> list[str]:
    """解析marker元数据并重算命令、合同及全部产物哈希。"""
    plan_by_name = {step.name: step for step in build_plan()}
    step = plan_by_name.get(name)
    if step is None:
        return [f"unknown marker step: {name}"]
    marker = paths().markers / f"{name}.done"
    if not marker.is_file():
        return [f"marker missing: {name}"]
    try:
        record = _json(marker)
    except (OSError, json.JSONDecodeError) as error:
        return [f"marker JSON invalid: {name}: {error}"]
    expected_contract = (
        "precontract:image_quality_v2"
        if name == "image_quality_v2"
        else load_contract(fresh=False)["contract_sha256"]
    )
    errors: list[str] = []
    if record.get("step") != name:
        errors.append(f"marker step mismatch: {name}")
    if record.get("contract_sha256") != expected_contract:
        errors.append(f"marker contract mismatch: {name}")
    if record.get("command") != _module_command(step.args):
        errors.append(f"marker command mismatch: {name}")
    current_hashes = _current_artifact_hashes(name)
    if not current_hashes or record.get("artifact_sha256") != current_hashes:
        errors.append(f"marker artifact hashes mismatch: {name}")
    if validate_artifact:
        errors.extend(validate_step(name, deep=True, contract_check=False))
    return errors


def _ci_errors(
    label: str,
    row: Any,
    expected_n: int,
    point_estimate: float | None = None,
) -> list[str]:
    """验证 bootstrap 行的样本数、有限性和区间闭合性。"""
    if not isinstance(row, dict):
        return [f"{label} CI row missing"]
    errors: list[str] = []
    if (
        row.get("n_boot") != expected_n
        or row.get("n_boot_requested") != expected_n
        or row.get("n_boot_effective") != expected_n
    ):
        errors.append(f"{label} bootstrap count != {expected_n}")
    try:
        mean = float(row["mean"])
        low = float(row["ci_lo"])
        high = float(row["ci_hi"])
        sd = float(row["sd"])
        stored_point = float(row["point_estimate"])
    except (KeyError, TypeError, ValueError):
        return [*errors, f"{label} CI numeric fields invalid"]
    if not all(np.isfinite(value) for value in (mean, low, high, sd, stored_point)):
        errors.append(f"{label} CI contains non-finite value")
    elif low > high or sd < 0:
        errors.append(f"{label} CI ordering or SD invalid")
    if point_estimate is not None:
        try:
            estimate = float(point_estimate)
        except (TypeError, ValueError):
            errors.append(f"{label} point estimate invalid")
        else:
            if not np.isfinite(estimate) or not np.isclose(
                estimate, stored_point, rtol=1e-9, atol=1e-12
            ):
                errors.append(f"{label} point estimate != recomputed value")
    expected_observations: int | None = None
    expected_clusters: int | None = None
    if label.startswith(
        (
            "final_ci/",
            "calibration_ci/",
            "per_class_ci/",
            "delta_vs_",
            "analysis/final_ci/",
            "analysis/calibration_ci/",
            "analysis/per_class_ci/",
            "analysis/delta_vs_",
        )
    ):
        expected_observations = 12_102
        expected_clusters = EXPECTED_TEST_PATIENTS
    elif label.startswith(("headline/biopsy/", "patient_level/biopsy_")):
        expected_observations = EXPECTED_TEST_BIOPSIES
        expected_clusters = EXPECTED_TEST_PATIENTS
    elif label.startswith(("headline/patient/", "patient_level/patient_")):
        expected_observations = EXPECTED_ELIGIBLE_PATIENTS
        expected_clusters = EXPECTED_ELIGIBLE_PATIENTS
    if expected_observations is not None and (
        row.get("bootstrap_unit") != "patient cluster"
        or row.get("n_observations") != expected_observations
        or row.get("n_clusters") != expected_clusters
    ):
        errors.append(
            f"{label} bootstrap audit unit/count mismatch "
            f"({expected_observations}/{expected_clusters})"
        )
    return errors


def _finite_metric_errors(label: str, value: Any, low: float, high: float) -> list[str]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return [f"{label} is not numeric"]
    if not np.isfinite(number) or not low <= number <= high:
        return [f"{label} outside finite range [{low}, {high}]"]
    return []


def _full_metrics_errors(
    label: str, value: Any, classes: list[str] | tuple[str, ...]
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} metrics missing"]
    errors: list[str] = []
    unit_metrics = {
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "macro_sensitivity",
        "macro_specificity",
        "worst_class_recall",
        "auc_ovr_macro",
        "macro_ap",
    }
    for key in unit_metrics:
        errors.extend(_finite_metric_errors(f"{label}/{key}", value.get(key), 0, 1))
    for key in ("mcc", "cohen_kappa"):
        errors.extend(_finite_metric_errors(f"{label}/{key}", value.get(key), -1, 1))
    per_class = value.get("per_class")
    if not isinstance(per_class, dict) or set(per_class) != set(classes):
        return [*errors, f"{label}/per_class class set mismatch"]
    for class_name in classes:
        row = per_class[class_name]
        if not isinstance(row, dict) or not {
            "n",
            "recall",
            "precision",
            "f1",
            "ap",
        }.issubset(row):
            errors.append(f"{label}/per_class/{class_name} schema invalid")
            continue
        if not isinstance(row["n"], int) or row["n"] <= 0:
            errors.append(f"{label}/per_class/{class_name}/n invalid")
        for key in ("recall", "precision", "f1", "ap"):
            errors.extend(
                _finite_metric_errors(
                    f"{label}/per_class/{class_name}/{key}", row[key], 0, 1
                )
            )
    return errors


def _analysis_step_errors(value: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    """不依赖后续步骤的 analysis 单步语义深验。"""
    errors: list[str] = []
    required = {
        "singles",
        "ensembles",
        "tta_sweep",
        "fusion_selection_protocol",
        "best_single_selection",
        "calibration",
        "final_ci",
        "calibration_ci",
        "per_class_ci",
        "delta_vs_best_single_ci",
        "delta_vs_resnet50_ci",
        "headline_aggregation",
        "patient_level",
        "inference_cost",
        "nonlinear_stacker_sensitivity",
        "primary_lr_stacker_fit",
        "voting_weights",
        "final_key",
        "selection_on_metacal",
        "confusion_matrix",
        "pr_curves",
        "dca",
        "diversity",
        "referral",
    }
    if not required.issubset(value):
        return [f"analysis semantic fields missing: {sorted(required - set(value))}"]
    best = value.get("best_single_selection", {})
    scores = best.get("train_oof_tta1_macro_f1", {})
    if (
        set(scores) != set(contract["backbones"])
        or best.get("selected") not in contract["backbones"]
        or best.get("selection_source") != "complete original-Train OOF TTA1"
        or best.get("metacal_labels_used_for_selection") is not False
        or best.get("test_labels_used_for_selection") is not False
    ):
        errors.append("analysis best-single selection semantic mismatch")
    weights = value.get("voting_weights", {})
    try:
        weight_values = np.asarray(
            [float(weights[name]) for name in contract["backbones"]]
        )
        if (
            set(weights) != set(contract["backbones"])
            or np.any(~np.isfinite(weight_values))
            or np.any(weight_values < 0)
            or not np.isclose(weight_values.sum(), 1.0, atol=1e-9)
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        errors.append("analysis voting weights invalid")
    selection = value.get("selection_on_metacal", {})
    expected_candidates = ["voting_tta8", "stack_lr_tta8"]
    try:
        scores_selection = {
            name: float(selection[name]["macro_f1"]) for name in expected_candidates
        }
        chosen = max(
            expected_candidates,
            key=lambda name: (
                scores_selection[name],
                -expected_candidates.index(name),
            ),
        )
        if (
            set(selection) != set(expected_candidates)
            or value.get("final_key") != chosen
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        errors.append("analysis Meta-Cal candidate selection invalid")
    primary_fit = value.get("primary_lr_stacker_fit", {})
    if not (
        primary_fit.get("converged") is True
        and isinstance(primary_fit.get("n_iter"), int)
        and 0 < primary_fit["n_iter"] < primary_fit.get("max_iter", 0)
        and primary_fit.get("feature_dim") == len(contract["backbones"]) * NUM_CLASSES
        and primary_fit.get("solver") == "lbfgs"
        and primary_fit.get("C") == 1.0
        and re.fullmatch(r"[0-9a-f]{64}", str(primary_fit.get("parameter_sha256", "")))
    ):
        errors.append("analysis primary LR convergence metadata invalid")
    temperature_fit = value.get("calibration", {}).get("temperature_fit", {})
    try:
        if not (
            value["calibration"]["temperature"] > 0
            and temperature_fit.get("success") is True
            and temperature_fit.get("bounds") == [0.05, 10.0]
            and int(temperature_fit.get("nfev", 0)) > 0
            and np.isfinite(float(temperature_fit["objective_nll"]))
            and np.isfinite(float(temperature_fit["identity_nll"]))
            and float(temperature_fit["objective_nll"])
            <= float(temperature_fit["identity_nll"]) + 1e-10
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        errors.append("analysis temperature optimizer metadata invalid")
    for level in ("biopsy", "patient"):
        if (
            value.get("headline_aggregation", {})
            .get(level, {})
            .get("mean_probability_source")
            != "globally temperature-calibrated Test probabilities"
        ):
            errors.append(f"analysis {level} mean aggregation source mismatch")
    confusion = value.get("confusion_matrix", {})
    try:
        counts = np.asarray(confusion["counts"])
        normalized = np.asarray(confusion["row_normalized"], dtype=float)
        if (
            counts.shape != (NUM_CLASSES, NUM_CLASSES)
            or normalized.shape != counts.shape
            or np.any(counts < 0)
            or np.any(counts != counts.astype(int))
            or int(counts.sum()) != int(contract["test_images"])
            or np.any(~np.isfinite(normalized))
            or not np.allclose(normalized.sum(axis=1), 1.0, atol=2e-5)
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        errors.append("analysis confusion matrix invalid")
    for count in (1, 2, 4, 8, 16):
        row = value.get("tta_sweep", {}).get(str(count), {})
        for method in ("voting", "stack_lr"):
            if not isinstance(row.get(method), dict) or "macro_f1" not in row[method]:
                errors.append(f"analysis TTA{count} {method} unavailable")
        try:
            timing = float(row["ms_per_image_ensemble"])
            if not np.isfinite(timing) or timing <= 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            errors.append(f"analysis TTA{count} measured timing invalid")
    final_ci_metrics = {
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "cohen_kappa",
        "auc_ovr_macro",
        "macro_sensitivity",
        "macro_specificity",
        "macro_ap",
        "worst_class_recall",
    }
    if set(value.get("final_ci", {})) != final_ci_metrics:
        errors.append("analysis final_ci metric set mismatch")
    for metric in final_ci_metrics:
        errors.extend(
            _ci_errors(
                f"analysis/final_ci/{metric}",
                value.get("final_ci", {}).get(metric),
                BOOTSTRAP_EXPECTED,
            )
        )
    for phase in ("before", "after"):
        rows = value.get("calibration_ci", {}).get(phase, {})
        if set(rows) != {"ece", "brier_multiclass", "nll"}:
            errors.append(f"analysis calibration_ci/{phase} mismatch")
        for metric in ("ece", "brier_multiclass", "nll"):
            errors.extend(
                _ci_errors(
                    f"analysis/calibration_ci/{phase}/{metric}",
                    rows.get(metric),
                    BOOTSTRAP_EXPECTED,
                )
            )
    per_class = value.get("per_class_ci", {})
    if set(per_class) != set(contract["classes"]):
        errors.append("analysis per_class_ci class set mismatch")
    for class_name in contract["classes"]:
        rows = per_class.get(class_name, {})
        if set(rows) != {"precision", "recall", "f1", "ap"}:
            errors.append(f"analysis per_class_ci/{class_name} metric set mismatch")
        for metric in ("precision", "recall", "f1", "ap"):
            errors.extend(
                _ci_errors(
                    f"analysis/per_class_ci/{class_name}/{metric}",
                    rows.get(metric),
                    500,
                )
            )
    paired_metrics = {
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "macro_ap",
        "worst_class_recall",
        *(f"recall_{name}" for name in contract["classes"]),
    }
    for key in ("delta_vs_best_single_ci", "delta_vs_resnet50_ci"):
        rows = value.get(key, {}).get("deltas", {})
        if set(rows) != paired_metrics:
            errors.append(f"analysis {key} metric set mismatch")
        for metric in paired_metrics:
            errors.extend(
                _ci_errors(
                    f"analysis/{key}/{metric}",
                    rows.get(metric),
                    BOOTSTRAP_EXPECTED,
                )
            )
    classes = list(contract["classes"])
    pr_curves = value.get("pr_curves", {})
    if not isinstance(pr_curves, dict) or set(pr_curves) != set(classes):
        errors.append("analysis PR curve class set mismatch")
    else:
        for class_name, row in pr_curves.items():
            try:
                precision = np.asarray(row["precision"], dtype=float)
                recall = np.asarray(row["recall"], dtype=float)
                thresholds = np.asarray(row["thresholds"], dtype=float)
                if (
                    precision.ndim != 1
                    or recall.ndim != 1
                    or thresholds.ndim != 1
                    or len(precision) != len(recall)
                    or len(precision) != len(thresholds) + 1
                    or len(thresholds) == 0
                    or np.any(~np.isfinite(precision))
                    or np.any(~np.isfinite(recall))
                    or np.any(~np.isfinite(thresholds))
                    or np.any((precision < 0) | (precision > 1))
                    or np.any((recall < 0) | (recall > 1))
                    or np.any((thresholds < 0) | (thresholds > 1))
                    or np.any(np.diff(thresholds) < 0)
                    or np.any(np.diff(recall) > 1e-8)
                ):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                errors.append(f"analysis PR curve invalid: {class_name}")
    dca = value.get("dca", {})
    expected_dca_thresholds = np.linspace(0.05, 0.60, 23).round(3)
    if not isinstance(dca, dict) or set(dca) != set(classes):
        errors.append("analysis DCA class set mismatch")
    else:
        for class_name, payload in dca.items():
            try:
                prevalence = float(payload["prevalence"])
                rows = payload["curve"]
                thresholds = np.asarray([row["pt"] for row in rows], dtype=float)
                model_nb = np.asarray([row["nb_model"] for row in rows], dtype=float)
                treat_all = np.asarray([row["nb_treat_all"] for row in rows], dtype=float)
                treat_none = np.asarray([row["nb_treat_none"] for row in rows], dtype=float)
                expected_all = prevalence - (1 - prevalence) * thresholds / (
                    1 - thresholds
                )
                if (
                    not 0 <= prevalence <= 1
                    or len(rows) != 23
                    or not np.array_equal(thresholds, expected_dca_thresholds)
                    or np.any(~np.isfinite(model_nb))
                    or np.any(~np.isfinite(treat_all))
                    or np.any(~np.isfinite(treat_none))
                    or np.any(np.abs(model_nb) > 2)
                    or not np.allclose(treat_all, expected_all, atol=1e-12)
                    or not np.array_equal(treat_none, np.zeros(23))
                ):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                errors.append(f"analysis DCA invalid: {class_name}")
    diversity = value.get("diversity", {})
    expected_pairs = {
        f"{first}|{second}"
        for index, first in enumerate(contract["backbones"])
        for second in contract["backbones"][index + 1 :]
    }
    diversity_ranges = {
        "prob_correlation": (-1.0, 1.0),
        "disagreement": (0.0, 1.0),
        "kappa": (-1.0, 1.0),
        "mutual_info": (0.0, float(np.log(NUM_CLASSES))),
        "both_wrong_rate": (0.0, 1.0),
    }
    if not isinstance(diversity, dict) or set(diversity) != set(diversity_ranges):
        errors.append("analysis diversity metric set mismatch")
    else:
        for metric, (low, high) in diversity_ranges.items():
            rows = diversity[metric]
            if not isinstance(rows, dict) or set(rows) != expected_pairs:
                errors.append(f"analysis diversity pair set mismatch: {metric}")
                continue
            for pair, number in rows.items():
                errors.extend(
                    _finite_metric_errors(
                        f"analysis/diversity/{metric}/{pair}", number, low, high
                    )
                )
    referral = value.get("referral", {})
    referral_rows = referral.get("thresholds", {}) if isinstance(referral, dict) else {}
    expected_thresholds = [float(number) for number in contract["referral_thresholds"]]
    expected_keys = {str(number) for number in expected_thresholds}
    metacal_totals = referral.get("metacal_class_totals", {})
    test_totals = referral.get("test_class_totals", {})
    metacal_totals_valid = (
        isinstance(metacal_totals, dict)
        and set(metacal_totals) == set(classes)
        and all(isinstance(number, int) and number > 0 for number in metacal_totals.values())
        and sum(metacal_totals.values()) == int(contract["metacal_images"])
    )
    test_totals_valid = (
        isinstance(test_totals, dict)
        and set(test_totals) == set(classes)
        and all(isinstance(number, int) and number > 0 for number in test_totals.values())
        and sum(test_totals.values()) == int(contract["test_images"])
    )
    if (
        referral.get("selection_partition") != "Meta-Cal"
        or not isinstance(referral_rows, dict)
        or set(referral_rows) != expected_keys
        or not metacal_totals_valid
        or not test_totals_valid
    ):
        errors.append("analysis referral top-level/count contract mismatch")
    else:
        eligible_thresholds: list[float] = []
        for threshold in expected_thresholds:
            row = referral_rows[str(threshold)]
            try:
                meta_total = int(row["metacal_n_total"])
                meta_auto = int(row["metacal_n_autonomous"])
                meta_referred = int(row["metacal_n_referred"])
                test_total = int(row["test_n_total"])
                test_auto = int(row["test_n_autonomous"])
                test_referred = int(row["test_n_referred"])
                meta_coverage = float(row["metacal_coverage"])
                test_coverage = float(row["test_coverage"])
                if (
                    meta_total != int(contract["metacal_images"])
                    or test_total != int(contract["test_images"])
                    or meta_auto + meta_referred != meta_total
                    or test_auto + test_referred != test_total
                    or not np.isclose(meta_coverage, meta_auto / meta_total, atol=1e-12)
                    or not np.isclose(test_coverage, test_auto / test_total, atol=1e-12)
                    or row["coverage"] != row["test_coverage"]
                    or row["auto_accuracy"] != row["test_autonomous_accuracy"]
                    or row["referred_accuracy"] != row["test_referred_accuracy"]
                ):
                    raise ValueError
                for prefix, totals, referred_count in (
                    ("metacal", metacal_totals, meta_referred),
                    ("test", test_totals, test_referred),
                ):
                    distribution = row[f"{prefix}_referred_class_dist"]
                    rates = row[f"{prefix}_per_class_referral_rate"]
                    if (
                        set(distribution) != set(classes)
                        or set(rates) != set(classes)
                        or any(not isinstance(number, int) or number < 0 for number in distribution.values())
                        or sum(distribution.values()) != referred_count
                        or any(
                            not np.isclose(
                                float(rates[name]), distribution[name] / totals[name], atol=1e-12
                            )
                            for name in classes
                        )
                    ):
                        raise ValueError
                if row["referred_class_dist"] != row["test_referred_class_dist"] or row[
                    "per_class_referral_rate"
                ] != row["test_per_class_referral_rate"]:
                    raise ValueError
                for key, count in (
                    ("metacal_auto_acc", meta_auto),
                    ("metacal_autonomous_accuracy", meta_auto),
                    ("auto_accuracy", test_auto),
                    ("test_autonomous_accuracy", test_auto),
                    ("referred_accuracy", test_referred),
                    ("test_referred_accuracy", test_referred),
                ):
                    score = row[key]
                    if (count == 0 and score is not None) or (
                        count > 0
                        and (score is None or not 0 <= float(score) <= 1)
                    ):
                        raise ValueError
                if row["metacal_auto_acc"] is not None and float(
                    row["metacal_auto_acc"]
                ) >= 0.90:
                    eligible_thresholds.append(threshold)
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                errors.append(f"analysis referral row invalid: {threshold}")
        expected_chosen = eligible_thresholds[0] if eligible_thresholds else None
        if referral.get("chosen_threshold") != expected_chosen:
            errors.append("analysis referral chosen threshold is not lowest qualifying")
    return errors


def _ablation_step_errors(
    value: dict[str, Any], contract: dict[str, Any]
) -> list[str]:
    """深验九臂 TTA1 统一 schema。"""
    expected = {
        "ce_plain",
        "ce_sampler",
        "focal_only",
        "focal_sampler",
        "full",
        "full_cbreweight",
        "cb_only",
        "mixup_only",
        "cutmix_only",
    }
    arms = value.get("arms")
    if not isinstance(arms, dict) or set(arms) != expected:
        return ["extended ablation arm set mismatch"]
    errors: list[str] = []
    required = {
        "flags",
        "metrics",
        "ece",
        "nll",
        "brier_multiclass",
        "minority_overcompensation_index",
        "pred_vs_true_share",
        "evaluation_tta",
    }
    expected_flags = {
        "ce_plain": (False, False, False, "none", False),
        "ce_sampler": (True, False, False, "none", False),
        "focal_only": (False, True, False, "none", False),
        "focal_sampler": (True, True, False, "none", False),
        "full": (True, True, True, "alternating", False),
        "full_cbreweight": (True, True, True, "alternating", True),
        "cb_only": (False, False, False, "none", True),
        "mixup_only": (False, False, True, "mixup", False),
        "cutmix_only": (False, False, True, "cutmix", False),
    }
    flag_keys = {"sampler", "focal", "mix", "mix_mode", "cb_reweight"}
    for arm, row in arms.items():
        if (
            not isinstance(row, dict)
            or not required.issubset(row)
            or row.get("evaluation_tta") != 1
        ):
            errors.append(f"extended ablation schema/TTA mismatch: {arm}")
            continue
        flags = row.get("flags", {})
        expected_tuple = expected_flags[arm]
        observed_tuple = (
            flags.get("sampler"),
            flags.get("focal"),
            flags.get("mix"),
            flags.get("mix_mode"),
            flags.get("cb_reweight"),
        ) if isinstance(flags, dict) else None
        if not isinstance(flags, dict) or set(flags) != flag_keys or observed_tuple != expected_tuple:
            errors.append(f"extended ablation flags mismatch: {arm}")
        errors.extend(
            _full_metrics_errors(
                f"extended_ablation/{arm}/metrics",
                row.get("metrics"),
                contract["classes"],
            )
        )
        errors.extend(
            _finite_metric_errors(
                f"extended_ablation/{arm}/ece", row.get("ece"), 0, 1
            )
        )
        errors.extend(
            _finite_metric_errors(
                f"extended_ablation/{arm}/nll", row.get("nll"), 0, 100
            )
        )
        errors.extend(
            _finite_metric_errors(
                f"extended_ablation/{arm}/brier_multiclass",
                row.get("brier_multiclass"),
                0,
                2,
            )
        )
        errors.extend(
            _finite_metric_errors(
                f"extended_ablation/{arm}/minority_overcompensation_index",
                row.get("minority_overcompensation_index"),
                -1,
                1,
            )
        )
        shares = row.get("pred_vs_true_share")
        if not isinstance(shares, dict) or set(shares) != set(contract["classes"]):
            errors.append(f"extended ablation class-share set mismatch: {arm}")
        else:
            for source in ("pred", "true"):
                try:
                    values = np.asarray(
                        [shares[name][source] for name in contract["classes"]],
                        dtype=float,
                    )
                    if (
                        np.any(~np.isfinite(values))
                        or np.any((values < 0) | (values > 1))
                        or not np.isclose(values.sum(), 1.0, atol=1e-9)
                    ):
                        raise ValueError
                except (KeyError, TypeError, ValueError):
                    errors.append(
                        f"extended ablation {source} class shares invalid: {arm}"
                    )
    return errors


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _archive_valid(path: Path, required_suffixes: tuple[str, ...]) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if not all(
                any(name.endswith(suffix) for name in names)
                for suffix in required_suffixes
            ):
                return False
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def _train_step_parts(name: str) -> tuple[str, str, int | None, str | None] | None:
    match = re.fullmatch(r"outer_train_(.+)_fold([0-4])", name)
    if match:
        return "outer", match.group(1), int(match.group(2)), None
    match = re.fullmatch(r"terminal_train_(.+)", name)
    if match:
        return "terminal", match.group(1), None, None
    match = re.fullmatch(r"ablation_train_(cb_only|mixup_only|cutmix_only)", name)
    if match:
        return "ablation", "", None, match.group(1)
    return None


def _prediction_step_parts(
    name: str,
) -> tuple[str, str, str, int | None, str | None] | None:
    match = re.fullmatch(r"outer_pred_(.+)_fold([0-4])", name)
    if match:
        return "outer", match.group(1), "train_oof", int(match.group(2)), None
    match = re.fullmatch(r"terminal_pred_(.+)_(metacal|test)", name)
    if match:
        return "terminal", match.group(1), match.group(2), None, None
    match = re.fullmatch(
        r"ablation_pred_(cb_only|mixup_only|cutmix_only)_(metacal|test)", name
    )
    if match:
        return "ablation", "", match.group(2), None, match.group(1)
    return None


def validate_contract_only() -> list[str]:
    """验证协议与静态执行计划。"""
    errors: list[str] = []
    try:
        contract = load_contract(fresh=True)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        return [f"contract: {error}"]
    plan = build_plan()
    if len(plan) != EXPECTED_TOTAL_STEPS:
        errors.append(f"plan step count={len(plan)}")
    if (
        sum(step.args[0] == "strict_oof/train.py" for step in plan)
        != EXPECTED_TRAINING_RUNS
    ):
        errors.append("GPU training run count is not 39")
    if contract["same_protocol_supervised_models"] != list(TRAIN_MODELS):
        errors.append("same-protocol model set mismatch")
    if contract["outer_folds"] != OUTER_FOLDS:
        errors.append("outer fold count mismatch")
    if contract["oof_tta_count"] != OOF_TTA_COUNT:
        errors.append("OOF TTA count mismatch")
    if contract["final_metacal_test_tta_count"] != FINAL_TTA_COUNT:
        errors.append("final TTA count mismatch")
    split_gate = contract.get("patient_split_gate", {})
    if (
        split_gate.get("total") != 8_250
        or split_gate.get("by_subset")
        != {"train": 5_940, "metacal": 661, "test": 1_649}
        or any(split_gate.get("pairwise_overlaps", {}).values())
    ):
        errors.append("patient split gate 8250=5940+661+1649 or overlap failed")
    qc = contract.get("qc_contract", {})
    if (
        qc.get("raw_images") != 60_512
        or qc.get("usable_images") != 60_509
        or qc.get("excluded_images") != 3
        or qc.get("by_subset") != {"train": 2, "metacal": 0, "test": 1}
        or len(qc.get("exclusions", [])) != 3
        or len({row.get("source_path_sha256") for row in qc.get("exclusions", [])}) != 3
    ):
        errors.append("QC 60512/60509/3 whitelist contract mismatch")
    p = paths()
    if _inside(p.out, p.exp / "results") is False or p.out == p.exp / "results":
        errors.append("strict result root is not isolated")
    if _inside(p.big, p.exp / "big") is False or p.big == p.exp / "big":
        errors.append("strict big root is not isolated")
    if set(contract["outer_assignment"].values()) != set(range(OUTER_FOLDS)):
        errors.append("outer assignments do not cover folds 0..4")
    for record in contract["fold_records"]:
        if any(
            record[key] != 0
            for key in ("outer_patient_overlap", "inner_patient_overlap")
        ):
            errors.append(f"fold {record['fold']} patient leakage")
        if not all(
            record[key]
            for key in (
                "outer_train_all_classes",
                "outer_heldout_all_classes",
                "inner_train_all_classes",
                "inner_validation_all_classes",
            )
        ):
            errors.append(f"fold {record['fold']} class coverage failure")
    return errors


def validate_step(
    name: str, deep: bool = False, contract_check: bool = True
) -> list[str]:
    """验证一个步骤的产物合同。"""
    if name == "image_quality_v2":
        try:
            validate_image_quality_v2()
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
            return [f"image-quality v2: {error}"]
        return []
    errors = validate_contract_only() if contract_check else []
    if errors:
        return errors
    contract = load_contract(fresh=False)
    p = paths()
    if name == "prepare_contract":
        return []
    train_parts = _train_step_parts(name)
    if train_parts:
        stage, model, fold, arm = train_parts
        if stage == "ablation":
            model = contract["additional_ablation_backbone"]
        checkpoint, _resume, log_path, _lock = artifact_paths(
            stage, model, fold=fold, arm=arm
        )
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 1_000_000:
            return [f"checkpoint missing/too small: {checkpoint}"]
        if not log_path.is_file():
            return [f"train log missing: {log_path}"]
        try:
            record = _json(log_path)
        except (OSError, json.JSONDecodeError) as error:
            return [f"train log invalid: {error}"]
        expected_run = (
            f"{stage}:{model}:fold={fold}"
            if stage != "ablation"
            else f"ablation:{arm}:{model}"
        )
        if record.get("status") != "complete":
            errors.append("train status is not complete")
        if record.get("contract_sha256") != contract["contract_sha256"]:
            errors.append("train contract mismatch")
        if record.get("run_key") != expected_run:
            errors.append("train run_key mismatch")
        expected_group_metadata = contract["runtime_preflight"][
            "pretrained_fingerprint_metadata"
        ][model]
        observed_group_metadata = record.get("optimizer_parameter_groups")
        group_keys = {
            "classifier_prefix",
            "classifier_parameter_count",
            "classifier_parameter_names_sha256",
            "backbone_parameter_count",
            "backbone_parameter_names_sha256",
        }
        if (
            not isinstance(observed_group_metadata, dict)
            or set(observed_group_metadata) != group_keys
            or any(
                observed_group_metadata.get(key) != expected_group_metadata.get(key)
                for key in group_keys
            )
        ):
            errors.append("train optimizer parameter-group partition mismatch")
        # 单步 marker 恢复也必须深验 checkpoint，不得只看大小/ZIP。
        if not _archive_valid(checkpoint, ("data.pkl", "version")):
            errors.append(f"checkpoint archive invalid: {checkpoint}")
        else:
            try:
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if not isinstance(state, dict):
                    raise ValueError("checkpoint 顶层非 dict")
                if state.get("contract_sha256") != contract["contract_sha256"]:
                    errors.append("checkpoint contract mismatch")
                if state.get("run_key") != expected_run:
                    errors.append("checkpoint run_key mismatch")
                if state.get("model_name") != record.get("model_name"):
                    errors.append("checkpoint model_name mismatch")
                expected_model_name = contract["model_architectures"].get(model)
                if state.get("model_name") != expected_model_name:
                    errors.append("checkpoint model architecture != contract")
                if not isinstance(state.get("model"), dict) or not state.get("model"):
                    errors.append("checkpoint model state missing")
                else:
                    import timm

                    semantic_model = timm.create_model(
                        expected_model_name,
                        pretrained=False,
                        num_classes=NUM_CLASSES,
                    )
                    semantic_model.load_state_dict(state["model"], strict=True)
                    if any(
                        (tensor.is_floating_point() or tensor.is_complex())
                        and not torch.isfinite(tensor).all()
                        for tensor in state["model"].values()
                        if isinstance(tensor, torch.Tensor)
                    ):
                        errors.append(
                            "checkpoint model state contains non-finite values"
                        )
                if int(state.get("epoch", -1)) != int(record.get("best_epoch", -2)):
                    errors.append("checkpoint epoch != train log best_epoch")
            except (OSError, RuntimeError, ValueError, EOFError) as error:
                errors.append(f"checkpoint semantic load failed: {error}")
        if stage == "terminal":
            outer_best_epochs: list[int] = []
            for outer_fold in range(OUTER_FOLDS):
                _outer_checkpoint, _outer_resume, outer_log, _outer_lock = (
                    artifact_paths("outer", model, fold=outer_fold)
                )
                try:
                    outer_record = _json(outer_log)
                    if (
                        outer_record.get("contract_sha256")
                        != contract["contract_sha256"]
                        or outer_record.get("status") != "complete"
                        or outer_record.get("monitor_partition") != "inner_validation"
                    ):
                        raise ValueError("outer training log contract/status mismatch")
                    epochs = outer_record.get("epochs")
                    if (
                        not isinstance(epochs, list)
                        or not epochs
                        or len(epochs) != int(outer_record["stopped_epoch"])
                    ):
                        raise ValueError("outer epoch history invalid")
                    outer_best_epochs.append(int(outer_record["best_epoch"]))
                except (
                    OSError,
                    ValueError,
                    TypeError,
                    KeyError,
                    json.JSONDecodeError,
                ) as error:
                    errors.append(f"terminal source outer fold {outer_fold}: {error}")
            if len(outer_best_epochs) == OUTER_FOLDS:
                expected_epoch = terminal_epoch(outer_best_epochs)
                if int(record.get("best_epoch", -1)) != expected_epoch:
                    errors.append("terminal epoch != current outer median")
                if record.get("outer_best_epochs") != outer_best_epochs:
                    errors.append("terminal outer_best_epochs lineage mismatch")
                if record.get("early_stopping") is not False:
                    errors.append("terminal early stopping must be false")
        return errors
    prediction_parts = _prediction_step_parts(name)
    if prediction_parts:
        stage, model, subset, fold, arm = prediction_parts
        if stage == "ablation":
            model = contract["additional_ablation_backbone"]
        bundle, _shards, meta, _lock = prediction_paths(
            stage, model, subset, fold=fold, arm=arm
        )
        expected_tta = OOF_TTA_COUNT if stage == "outer" else FINAL_TTA_COUNT
        expected_n = (
            contract["fold_records"][fold]["outer_heldout_images"]
            if stage == "outer"
            else contract[f"{subset}_images"]
        )
        if not bundle.is_file() or not meta.is_file():
            return [f"prediction bundle/meta missing: {bundle}"]
        checkpoint, _resume, _train_log, _train_lock = artifact_paths(
            stage, model, fold=fold, arm=arm
        )
        if not checkpoint.is_file():
            return [f"prediction source checkpoint missing: {checkpoint}"]
        checkpoint_hash = file_sha256(checkpoint)
        try:
            indexed_rows, expected_partition = _load_prediction_rows(
                stage, subset, fold, contract
            )
            expected_indices = np.asarray(
                [index for index, _row in indexed_rows], dtype=np.int64
            )
            expected_labels = np.asarray(
                [
                    contract["classes"].index(str(row["cls"]))
                    for _index, row in indexed_rows
                ],
                dtype=np.int64,
            )
            expected_patients = np.asarray(
                [str(row["pt"]) for _index, row in indexed_rows]
            )
            with np.load(bundle, allow_pickle=False) as archive:
                if archive["probs"].shape != (expected_tta, expected_n, NUM_CLASSES):
                    errors.append(
                        f"prediction shape mismatch: {archive['probs'].shape}"
                    )
                if archive["labels"].shape != (expected_n,):
                    errors.append("prediction labels shape mismatch")
                if archive["global_indices"].shape != (expected_n,):
                    errors.append("prediction indices shape mismatch")
                elif not np.array_equal(
                    archive["global_indices"].astype(np.int64), expected_indices
                ):
                    errors.append("prediction global_indices/source order mismatch")
                if not np.array_equal(
                    archive["labels"].astype(np.int64), expected_labels
                ):
                    errors.append("prediction labels/source order mismatch")
                if archive["patients"].shape != (expected_n,) or not np.array_equal(
                    archive["patients"].astype(str), expected_patients
                ):
                    errors.append("prediction patients/source order mismatch")
                if str(archive["source_partition"].item()) != expected_partition:
                    errors.append("prediction source_partition mismatch")
                if (
                    bool(archive["labels_used_for_model_or_selection"].item())
                    is not False
                ):
                    errors.append("prediction label-contact flag")
                if (
                    str(archive["contract_sha256"].item())
                    != contract["contract_sha256"]
                ):
                    errors.append("prediction contract mismatch")
                if str(archive["checkpoint_sha256"].item()) != checkpoint_hash:
                    errors.append("prediction checkpoint lineage mismatch")
                probs = archive["probs"].astype(np.float32)
                if np.any(~np.isfinite(probs)):
                    errors.append("prediction contains non-finite values")
                if np.any(probs < -0.002) or np.any(probs > 1.002):
                    errors.append("prediction probabilities outside tolerance")
                if np.max(np.abs(probs.sum(axis=2) - 1.0)) > 0.02:
                    errors.append("prediction row sums outside float16 tolerance")
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"prediction invalid: {error}")
        if not _archive_valid(bundle, ("probs.npy", "labels.npy")):
            errors.append(f"prediction archive CRC invalid: {bundle}")
        try:
            metadata = _json(meta)
            expected_run_model = model
            required_meta = {
                "status": "complete",
                "contract_sha256": contract["contract_sha256"],
                "checkpoint_sha256": checkpoint_hash,
                "stage": stage,
                "model": expected_run_model,
                "fold": fold,
                "arm": arm,
                "subset": subset,
                "source_partition": expected_partition,
                "n_images": expected_n,
                "tta_count": expected_tta,
                "labels_used_for_model_or_selection": False,
            }
            for key, expected in required_meta.items():
                if metadata.get(key) != expected:
                    errors.append(f"prediction meta mismatch: {key}")
            if metadata.get("bundle") != str(bundle):
                errors.append("prediction meta bundle path mismatch")
            if (
                not isinstance(metadata.get("tta_seconds"), list)
                or len(metadata["tta_seconds"]) != expected_tta
                or any(
                    not np.isfinite(float(value)) or float(value) <= 0
                    for value in metadata.get("tta_seconds", [])
                )
            ):
                errors.append("prediction meta tta_seconds invalid")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            errors.append(f"prediction meta invalid: {error}")
        return errors
    if name == "strict_gradcam":
        target = p.out / "gradcam_index.json"
    elif name == "strict_analysis":
        target = p.out / "analysis.json"
    elif name == "extended_ablation_analysis":
        target = p.out / "extended_ablation_analysis.json"
    elif name == "validate_strict_oof":
        target = p.validation_report
    elif name == "finalize_strict_oof":
        target = p.ready
    else:
        return [f"unknown step: {name}"]
    if not target.is_file():
        return [f"missing output: {target}"]
    try:
        value = _json(target)
    except (OSError, json.JSONDecodeError) as error:
        return [f"invalid JSON {target}: {error}"]
    if value.get("contract_sha256") != contract["contract_sha256"]:
        errors.append(f"contract mismatch: {target}")
    expected_status = "ready" if name == "finalize_strict_oof" else "complete"
    if value.get("status") != expected_status:
        errors.append(f"status mismatch: {target}")
    if name == "validate_strict_oof" and value.get("ok") is not True:
        errors.append("validation report ok != true")
    if name == "validate_strict_oof":
        current_ledger: dict[str, str] = {}
        for step in build_plan():
            if step.name in {"validate_strict_oof", "finalize_strict_oof"}:
                continue
            current_ledger.update(_current_artifact_hashes(step.name))
        if not current_ledger or value.get("artifact_sha256") != current_ledger:
            errors.append("validation report artifact ledger is stale/incomplete")
        marker_ledger = value.get("marker_sha256")
        current_markers = {
            f"results/strict_oof_v1/markers/{step.name}.done": file_sha256(
                p.markers / f"{step.name}.done"
            )
            for step in build_plan()
            if step.name not in {"validate_strict_oof", "finalize_strict_oof"}
            and (p.markers / f"{step.name}.done").is_file()
        }
        if marker_ledger != current_markers:
            errors.append("validation report prerequisite-marker ledger is stale")
    if name == "strict_gradcam":
        checkpoint, _resume, _log, _lock = artifact_paths("terminal", "convnext_base")
        bundle, _shards, _meta, _lock = prediction_paths(
            "terminal", "convnext_base", "test"
        )
        if not checkpoint.is_file() or value.get("checkpoint_sha256") != file_sha256(
            checkpoint
        ):
            errors.append("strict Grad-CAM checkpoint lineage mismatch")
        if not bundle.is_file() or value.get("prediction_bundle_sha256") != file_sha256(
            bundle
        ):
            errors.append("strict Grad-CAM prediction lineage mismatch")
        cases = value.get("cases", [])

        def forbidden_identifier_keys(current: Any) -> list[str]:
            found: list[str] = []
            if isinstance(current, dict):
                for key, nested in current.items():
                    if any(
                        word in str(key).lower()
                        for word in ("patient", "path", "source")
                    ):
                        found.append(str(key))
                    found.extend(forbidden_identifier_keys(nested))
            elif isinstance(current, list):
                for nested in current:
                    found.extend(forbidden_identifier_keys(nested))
            return found

        forbidden_payload_keys = forbidden_identifier_keys(value)
        if forbidden_payload_keys:
            errors.append(
                "strict Grad-CAM identifier-like keys exported: "
                f"{sorted(set(forbidden_payload_keys))}"
            )
        required = {
            "case_id",
            "kind",
            "selection_reason",
            "selection_mode",
            "fallback_used",
            "true",
            "pred",
            "conf",
            "annotation_strip_masked_px",
            "tta_component",
            "png_sha256",
        }
        correct_classes: set[str] = set()
        error_classes: set[str] = set()
        igan_error_classes: set[str] = set()
        fallback_rows = 0
        case_ids: set[str] = set()
        if not isinstance(cases, list) or len(cases) < 11:
            errors.append("strict Grad-CAM has fewer than 11 cases")
        else:
            for row in cases:
                if not isinstance(row, dict) or not required.issubset(row):
                    errors.append("strict Grad-CAM case schema incomplete")
                    continue
                forbidden = {
                    key
                    for key in row
                    if any(
                        word in key.lower() for word in ("patient", "path", "source")
                    )
                }
                if (
                    forbidden
                    or re.fullmatch(r"gc_[0-9]{3}", str(row.get("case_id"))) is None
                    or row.get("case_id") in case_ids
                    or row.get("annotation_strip_masked_px") != 28
                    or row.get("tta_component") != 0
                    or row.get("selection_mode")
                    != value.get("misclassification_selection_mode")
                    or bool(row.get("fallback_used"))
                    != bool(value.get("fallback_used"))
                ):
                    errors.append(
                        f"strict Grad-CAM privacy/protocol failure: {row.get('case_id')}"
                    )
                case_ids.add(str(row.get("case_id")))
                try:
                    confidence = float(row.get("conf"))
                    if (
                        row.get("true") not in contract["classes"]
                        or row.get("pred") not in contract["classes"]
                        or not np.isfinite(confidence)
                        or not 0 <= confidence <= 1
                    ):
                        raise ValueError
                except (TypeError, ValueError):
                    errors.append(
                        f"strict Grad-CAM class/conf invalid: {row.get('case_id')}"
                    )
                png = p.out / "gradcam" / f"{row.get('case_id')}.png"
                png_valid = False
                try:
                    if png.is_file() and file_sha256(png) == row.get("png_sha256"):
                        with Image.open(png) as image:
                            png_valid = (
                                image.format == "PNG"
                                and image.width > 0
                                and image.height > 0
                                and image.mode in {"RGB", "RGBA"}
                            )
                            image.verify()
                except (OSError, ValueError):
                    png_valid = False
                if not png_valid:
                    errors.append(f"strict Grad-CAM PNG invalid: {png}")
                if row.get("kind") == "correct" and row.get("true") == row.get("pred"):
                    if row.get("selection_reason") != (
                        "highest-confidence correct within truth class"
                    ):
                        errors.append("strict Grad-CAM correct selection reason mismatch")
                    correct_classes.add(str(row["true"]))
                elif row.get("kind") in {
                    "missed_as_IgAN",
                    "misclassified_fallback",
                } and row.get("true") != row.get("pred"):
                    if (
                        row.get("kind") == "missed_as_IgAN"
                        and row.get("pred") != "IgAN"
                    ):
                        errors.append("missed_as_IgAN row does not predict IgAN")
                    if row.get("kind") == "missed_as_IgAN":
                        igan_error_classes.add(str(row["true"]))
                        if row.get("selection_reason") != (
                            "high-confidence component error predicted as IgAN"
                        ):
                            errors.append("missed_as_IgAN selection reason mismatch")
                    else:
                        fallback_rows += 1
                        if row.get("selection_reason") != (
                            "deterministic high-confidence component-error fallback"
                        ):
                            errors.append("Grad-CAM fallback selection reason mismatch")
                    error_classes.add(str(row["true"]))
                else:
                    errors.append(
                        f"strict Grad-CAM kind/correctness mismatch: {row.get('case_id')}"
                    )
            mode = value.get("misclassification_selection_mode")
            if mode not in {"igan_attraction", "stratified_misclassification_fallback"}:
                errors.append("strict Grad-CAM selection mode invalid")
            if bool(value.get("fallback_used")) != (
                mode == "stratified_misclassification_fallback"
            ):
                errors.append("strict Grad-CAM fallback flag mismatch")
            if value.get("missed_as_IgAN_truth_class_count") != len(
                igan_error_classes
            ):
                errors.append("strict Grad-CAM IgAN truth-class count mismatch")
            if mode == "igan_attraction" and (
                len(igan_error_classes) < 2 or fallback_rows != 0
            ):
                errors.append("strict Grad-CAM IgAN-directed mode contract mismatch")
            if mode == "stratified_misclassification_fallback" and (
                len(igan_error_classes) >= 2
                or fallback_rows == 0
                or len(error_classes) < 2
            ):
                errors.append("strict Grad-CAM fallback mode contract mismatch")
            if len(correct_classes) < 6 or len(error_classes) < 2:
                errors.append("strict Grad-CAM class coverage insufficient")
        if value.get("model") != "convnext_base" or value.get("tta_component") != 0:
            errors.append("strict Grad-CAM is not terminal ConvNeXt-B TTA0")
    if name == "strict_analysis":
        from strict_oof.analyze import analysis_input_sha256

        try:
            if value.get("input_prediction_sha256") != analysis_input_sha256():
                errors.append("strict analysis prediction lineage mismatch")
        except (OSError, ValueError) as error:
            errors.append(f"strict analysis lineage unavailable: {error}")
        errors.extend(_analysis_step_errors(value, contract))
    if name == "extended_ablation_analysis":
        from strict_oof.analyze_ablation import input_prediction_sha256

        try:
            if value.get("input_prediction_sha256") != input_prediction_sha256(
                contract
            ):
                errors.append("extended ablation prediction lineage mismatch")
        except (OSError, ValueError) as error:
            errors.append(f"extended ablation lineage unavailable: {error}")
        errors.extend(_ablation_step_errors(value, contract))
    if name == "finalize_strict_oof":
        delivery_relative = value.get("delivery_project_relative_path")
        expected_delivery_relative = p.delivery.relative_to(p.exp.parent).as_posix()
        delivery = (p.exp.parent / str(delivery_relative)).resolve()
        if (
            delivery_relative != expected_delivery_relative
            or delivery != p.delivery.resolve()
            or not delivery.is_dir()
        ):
            errors.append("READY does not point to composite delivery results")
        expected_relative_paths = {
            "strict_results_project_relative_path": p.out,
            "strict_big_project_relative_path": p.big,
            "delivery_project_relative_path": p.delivery,
        }
        for key, target in expected_relative_paths.items():
            if value.get(key) != target.relative_to(p.exp.parent).as_posix():
                errors.append(f"READY portable path mismatch: {key}")
        for filename in (
            *DELIVERY_LEGACY_FILES,
            "image_quality.json",
            "analysis.json",
            "ablation_analysis.json",
        ):
            if not (delivery / filename).is_file():
                errors.append(f"delivery file missing: {filename}")
        for filename in ("ALL.done", "core_validation.done"):
            if not (delivery / "markers" / filename).is_file():
                errors.append(f"delivery marker missing: {filename}")
        active = p.exp / "results" / "strict_oof_ACTIVE.json"
        try:
            if _json(active) != value:
                errors.append("ACTIVE pointer differs from READY")
        except (OSError, json.JSONDecodeError):
            errors.append("ACTIVE pointer missing/invalid")
        expected_ready_hashes = {
            "validation_report_sha256": p.validation_report,
            "analysis_sha256": p.out / "analysis.json",
            "extended_ablation_analysis_sha256": (
                p.out / "extended_ablation_analysis.json"
            ),
            "delivery_core_validation_sha256": delivery / "core_validation.json",
            "delivery_analysis_sha256": delivery / "analysis.json",
            "delivery_payload_manifest_sha256": delivery / "payload_manifest.json",
        }
        for key, current in expected_ready_hashes.items():
            if not current.is_file() or value.get(key) != file_sha256(current):
                errors.append(f"READY hash mismatch: {key}")
        manifest_path = delivery / "payload_manifest.json"
        try:
            manifest = _json(manifest_path)
            manifest_files = manifest.get("files", {})
            actual_files = {
                target.relative_to(delivery).as_posix()
                for target in delivery.rglob("*")
                if target.is_file() and target != manifest_path
            }
            from strict_oof.delivery import _expected_delivery_files

            expected_allowlist = _expected_delivery_files(
                delivery, contract, include_core=True
            )
            if (
                manifest.get("contract_sha256") != contract["contract_sha256"]
                or not isinstance(manifest_files, dict)
                or set(manifest_files) != actual_files
                or actual_files != expected_allowlist
            ):
                errors.append("delivery manifest file set/contract mismatch")
            else:
                for relative, expected in manifest_files.items():
                    if file_sha256(delivery / relative) != expected:
                        errors.append(f"delivery manifest hash mismatch: {relative}")
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            errors.append("delivery payload manifest invalid")
        try:
            core = _json(delivery / "core_validation.json")
            if (
                core.get("ok") is not True
                or core.get("errors")
                or core.get("contract_sha256")
                not in {
                    None,
                    contract["contract_sha256"],
                }
            ):
                errors.append("delivery core validation report is not green")
            core_marker = (delivery / "markers" / "core_validation.done").read_text(
                encoding="utf-8"
            )
            if (
                f"report_sha256={file_sha256(delivery / 'core_validation.json')}"
                not in core_marker
                or f"analysis_sha256={file_sha256(delivery / 'analysis.json')}"
                not in core_marker
                or f"contract_sha256={contract['contract_sha256']}" not in core_marker
            ):
                errors.append("delivery core marker/report binding mismatch")
        except (OSError, json.JSONDecodeError):
            errors.append("delivery core report/marker invalid")
    return errors


def validate_full() -> dict[str, Any]:
    """对所有高成本产物进行深验收。"""
    errors = validate_contract_only()
    checks: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    if errors:
        return {
            "status": "complete",
            "ok": False,
            "contract_sha256": None,
            "errors": errors,
            "checks": checks,
            "artifact_sha256": hashes,
        }
    contract = load_contract(fresh=True)
    p = paths()
    for relative, expected in contract["protected_legacy_sha256"].items():
        target = p.exp / relative
        if not target.is_file() or file_sha256(target) != expected:
            errors.append(f"protected legacy artifact changed: {relative}")
    legacy_root = p.exp / "results"
    current_legacy = {}
    for target in sorted(legacy_root.rglob("*"), key=lambda item: item.as_posix()):
        if not target.is_file():
            continue
        relative = target.relative_to(legacy_root)
        if relative.parts and relative.parts[0] == "strict_oof_v1":
            continue
        if relative.as_posix() == "strict_oof_ACTIVE.json":
            continue
        current_legacy[relative.as_posix()] = file_sha256(target)
    if current_legacy != contract.get("protected_legacy_tree_sha256"):
        errors.append("legacy results tree changed after strict contract freeze")
    plan = build_plan()
    prerequisite_steps = [
        step
        for step in plan
        if step.name not in {"validate_strict_oof", "finalize_strict_oof"}
    ]
    failed_markers = sorted(p.markers.glob("*.FAILED"))
    if failed_markers:
        errors.append(f"FAILED markers: {[path.name for path in failed_markers]}")
    for step in prerequisite_steps:
        marker_errors = validate_marker(step.name, validate_artifact=True)
        errors.extend(f"{step.name}: {error}" for error in marker_errors)
        if marker_errors:
            continue
        parts = _train_step_parts(step.name)
        if parts:
            stage, model, fold, arm = parts
            if stage == "ablation":
                model = contract["additional_ablation_backbone"]
            checkpoint, _resume, log, _lock = artifact_paths(
                stage, model, fold=fold, arm=arm
            )
            if checkpoint.is_file():
                hashes[str(checkpoint.relative_to(p.exp)).replace("\\", "/")] = (
                    file_sha256(checkpoint)
                )
            if log.is_file():
                hashes[str(log.relative_to(p.exp)).replace("\\", "/")] = file_sha256(
                    log
                )
        pred_parts = _prediction_step_parts(step.name)
        if pred_parts:
            stage, model, subset, fold, arm = pred_parts
            if stage == "ablation":
                model = contract["additional_ablation_backbone"]
            bundle, _shards, meta, _lock = prediction_paths(
                stage, model, subset, fold=fold, arm=arm
            )
            if bundle.is_file():
                hashes[str(bundle.relative_to(p.exp)).replace("\\", "/")] = file_sha256(
                    bundle
                )
            if meta.is_file():
                hashes[str(meta.relative_to(p.exp)).replace("\\", "/")] = file_sha256(
                    meta
                )
        if step.name == "strict_gradcam":
            index_path = p.out / "gradcam_index.json"
            if index_path.is_file():
                hashes[str(index_path.relative_to(p.exp)).replace("\\", "/")] = (
                    file_sha256(index_path)
                )
                for row in _json(index_path).get("cases", []):
                    png = p.out / "gradcam" / f"{row.get('case_id')}.png"
                    if png.is_file():
                        hashes[str(png.relative_to(p.exp)).replace("\\", "/")] = (
                            file_sha256(png)
                        )

    for model in TRAIN_MODELS:
        best_epochs = []
        for fold in range(OUTER_FOLDS):
            _checkpoint, _resume, log, _lock = artifact_paths("outer", model, fold=fold)
            if not log.is_file():
                continue
            row = _json(log)
            best_epochs.append(int(row["best_epoch"]))
            if row.get("monitor_partition") != "inner_validation":
                errors.append(f"{model}/fold{fold} did not use inner validation")
            if (
                row.get(
                    "outer_heldout_labels_used_for_gradient_early_stop_or_epoch_selection"
                )
                is not False
            ):
                errors.append(f"{model}/fold{fold} outer label-contact flag")
            if (
                len(row.get("class_counts_train", [])) != NUM_CLASSES
                or min(row.get("class_counts_train", [0])) <= 0
            ):
                errors.append(f"{model}/fold{fold} train class coverage")
        if len(best_epochs) == OUTER_FOLDS:
            _checkpoint, _resume, terminal_log, _lock = artifact_paths(
                "terminal", model
            )
            row = _json(terminal_log)
            expected_epoch = terminal_epoch(best_epochs)
            if row.get("best_epoch") != expected_epoch:
                errors.append(f"{model} terminal epoch != median outer best")
            if row.get("early_stopping") is not False:
                errors.append(f"{model} terminal early stopping was enabled")
            if row.get("scheduler_horizon_epochs") != 40:
                errors.append(f"{model} terminal scheduler horizon != 40")

    train_rows, _metacal_rows, test_rows, _input_hashes = source_rows()
    for model in TRAIN_MODELS:
        seen = np.zeros(len(train_rows), dtype=np.int8)
        for fold in range(OUTER_FOLDS):
            bundle, _shards, _meta, _lock = prediction_paths(
                "outer", model, "train_oof", fold=fold
            )
            if not bundle.is_file():
                continue
            with np.load(bundle, allow_pickle=False) as archive:
                indices = archive["global_indices"].astype(int)
                if np.any(indices < 0) or np.any(indices >= len(train_rows)):
                    errors.append(f"{model} outer indices out of range")
                elif np.any(seen[indices]):
                    errors.append(f"{model} duplicate Train OOF coverage")
                else:
                    seen[indices] = 1
        if np.any(seen != 1):
            errors.append(f"{model} incomplete Train OOF coverage")

    analysis_path = p.out / "analysis.json"
    if analysis_path.is_file():
        analysis = _json(analysis_path)
        required_analysis_fields = {
            "singles",
            "voting_weights",
            "tta_sweep",
            "ensembles",
            "final_key",
            "selection_on_metacal",
            "fusion_selection_protocol",
            "calibration",
            "classwise_temperature_probe",
            "final_ci",
            "calibration_ci",
            "per_class_ci",
            "diversity",
            "referral",
            "best_single_selection",
            "mcnemar_final_vs_best_single",
            "delta_vs_best_single_ci",
            "delta_vs_resnet50_ci",
            "confusion_matrix",
            "pr_curves",
            "patient_level",
            "dca",
            "inference_cost",
            "headline_aggregation",
            "nonlinear_stacker_sensitivity",
        }
        missing_analysis_fields = required_analysis_fields - set(analysis)
        if missing_analysis_fields:
            errors.append(
                f"analysis replacement fields missing: {sorted(missing_analysis_fields)}"
            )
        protocol = analysis.get("protocol", {})
        if protocol.get("test_labels_used_for_any_selection") is not False:
            errors.append("analysis test selection flag")
        if analysis.get("fusion_selection", {}).get("test_labels_used") is not False:
            errors.append("fusion selection used Test")
        expected_sensitivity = {"stack_mlp_tta1", "stack_gbdt_tta1"}
        fusion_protocol = analysis.get("fusion_selection_protocol", {})
        if (
            set(fusion_protocol.get("post_hoc_terminal_test_sensitivity", []))
            != expected_sensitivity
            or any(
                fusion_protocol.get("sensitivity_roles", {})
                .get(name, {})
                .get("used_for_fusion_selection")
                is not False
                for name in expected_sensitivity
            )
            or not expected_sensitivity.issubset(analysis.get("ensembles", {}))
        ):
            errors.append("MLP/GBDT post-hoc sensitivity contract incomplete")
        if (
            analysis.get("same_protocol_resnet50", {}).get("excluded_from_ensemble")
            is not True
        ):
            errors.append("ResNet50 ensemble exclusion missing")
        final_key = analysis.get("final_key")
        final_metrics = analysis.get("ensembles", {}).get(final_key, {})
        expected_final_ci = {
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "mcc",
            "cohen_kappa",
            "auc_ovr_macro",
            "macro_sensitivity",
            "macro_specificity",
            "macro_ap",
            "worst_class_recall",
        }
        final_ci = analysis.get("final_ci", {})
        if set(final_ci) != expected_final_ci:
            errors.append("final_ci metric set is not the prespecified 10")
        for metric in expected_final_ci:
            errors.extend(
                _ci_errors(
                    f"final_ci/{metric}",
                    final_ci.get(metric),
                    BOOTSTRAP_EXPECTED,
                    final_metrics.get(metric),
                )
            )
        calibration = analysis.get("calibration", {})
        calibration_ci = analysis.get("calibration_ci", {})
        for phase in ("before", "after"):
            if set(calibration_ci.get(phase, {})) != {
                "ece",
                "brier_multiclass",
                "nll",
            }:
                errors.append(f"calibration_ci/{phase} metric set mismatch")
            for metric in ("ece", "brier_multiclass", "nll"):
                errors.extend(
                    _ci_errors(
                        f"calibration_ci/{phase}/{metric}",
                        calibration_ci.get(phase, {}).get(metric),
                        BOOTSTRAP_EXPECTED,
                        calibration.get(phase, {}).get(metric),
                    )
                )
        class_ece = (
            analysis.get("calibration", {})
            .get("test_after", {})
            .get("one_vs_rest_ece", {})
        )
        if set(class_ece) != set(contract["classes"]):
            errors.append("standard OvR classwise ECE incomplete")
        if set(analysis.get("per_class_ci", {})) != set(contract["classes"]):
            errors.append("per-class CI class set incomplete")
        else:
            for class_name, metrics in analysis["per_class_ci"].items():
                if set(metrics) != {"precision", "recall", "f1", "ap"}:
                    errors.append(f"per-class CI metrics incomplete: {class_name}")
                for metric in ("precision", "recall", "f1", "ap"):
                    errors.extend(
                        _ci_errors(
                            f"per_class_ci/{class_name}/{metric}",
                            metrics.get(metric),
                            500,
                            final_metrics.get("per_class", {})
                            .get(class_name, {})
                            .get(metric),
                        )
                    )
        paired = analysis.get("delta_vs_best_single_ci", {}).get("deltas", {})
        if (
            not {
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
                "worst_class_recall",
            }.issubset(paired)
            or not all(f"recall_{name}" in paired for name in contract["classes"])
            or "macro_ap" not in paired
        ):
            errors.append("paired patient bootstrap deltas incomplete")
        paired_metric_names = {
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "macro_ap",
            "worst_class_recall",
            *(f"recall_{name}" for name in contract["classes"]),
        }
        best_selection = analysis.get("best_single_selection", {})
        best_name = best_selection.get("selected")
        best_scores = best_selection.get("train_oof_tta1_macro_f1", {})
        if (
            set(best_scores) != set(contract["backbones"])
            or best_name not in contract["backbones"]
            or best_name
            != max(
                contract["backbones"], key=lambda name: best_scores.get(name, -np.inf)
            )
            or best_selection.get("selection_source")
            != "complete original-Train OOF TTA1"
            or best_selection.get("criterion")
            != "highest complete Train OOF macro-F1 at TTA1"
            or best_selection.get("test_labels_used_for_selection") is not False
            or best_selection.get("metacal_labels_used_for_selection") is not False
        ):
            errors.append("best-single Train OOF TTA1 selection contract mismatch")
        best_metrics = analysis.get("singles", {}).get(best_name, {}).get("tta1", {})
        resnet_metrics = analysis.get("same_protocol_resnet50", {}).get("test", {})
        for comparison_key, comparator in (
            ("delta_vs_best_single_ci", best_metrics),
            ("delta_vs_resnet50_ci", resnet_metrics),
        ):
            deltas = analysis.get(comparison_key, {}).get("deltas", {})
            if set(deltas) != paired_metric_names:
                errors.append(f"{comparison_key} metric set mismatch")
            for metric in paired_metric_names:
                if metric.startswith("recall_"):
                    class_name = metric.removeprefix("recall_")
                    estimate = final_metrics.get("per_class", {}).get(
                        class_name, {}
                    ).get("recall", float("nan")) - comparator.get("per_class", {}).get(
                        class_name, {}
                    ).get(
                        "recall", float("nan")
                    )
                else:
                    estimate = final_metrics.get(metric, float("nan")) - comparator.get(
                        metric, float("nan")
                    )
                errors.extend(
                    _ci_errors(
                        f"{comparison_key}/{metric}",
                        deltas.get(metric),
                        BOOTSTRAP_EXPECTED,
                        estimate,
                    )
                )
        if set(analysis.get("pr_curves", {})) != set(contract["classes"]):
            errors.append("PR curves incomplete")
        if set(analysis.get("dca", {})) != set(contract["classes"]):
            errors.append("DCA incomplete")
        if len(analysis.get("diversity", {}).get("disagreement", {})) != 10:
            errors.append("five-backbone diversity matrix incomplete")
        referral_rows = analysis.get("referral", {}).get("thresholds", {})
        if set(referral_rows) != {
            str(value) for value in contract["referral_thresholds"]
        }:
            errors.append("referral threshold grid incomplete")
        else:
            required_referral = {
                "metacal_coverage",
                "metacal_auto_acc",
                "coverage",
                "auto_accuracy",
                "referred_accuracy",
                "referred_class_dist",
                "per_class_referral_rate",
            }
            for threshold, row in referral_rows.items():
                if not required_referral.issubset(row):
                    errors.append(f"referral fields incomplete at {threshold}")
        headline = analysis.get("headline_aggregation", {})
        biopsy = headline.get("biopsy", {})
        patient = headline.get("patient", {})
        actual_biopsies = len({str(row["pt"]) for row in test_rows})
        patient_labels: dict[str, set[str]] = {}
        for row in test_rows:
            patient_labels.setdefault(patient_of(row["pt"]), set()).add(row["cls"])
        actual_eligible = sum(len(labels) == 1 for labels in patient_labels.values())
        actual_mixed = sum(len(labels) > 1 for labels in patient_labels.values())
        if actual_biopsies != EXPECTED_TEST_BIOPSIES:
            errors.append(f"source Test biopsy count {actual_biopsies} != 1657")
        if actual_eligible != EXPECTED_ELIGIBLE_PATIENTS:
            errors.append(f"source eligible patient count {actual_eligible} != 1642")
        if len(patient_labels) != EXPECTED_TEST_PATIENTS:
            errors.append(f"source Test patient count {len(patient_labels)} != 1649")
        if actual_mixed != EXPECTED_MIXED_PATIENTS:
            errors.append(f"source mixed patient count {actual_mixed} != 7")
        if biopsy.get("n_evaluated") != EXPECTED_TEST_BIOPSIES:
            errors.append("headline biopsy n != 1657")
        if patient.get("n_evaluated") != EXPECTED_ELIGIBLE_PATIENTS:
            errors.append("headline eligible patient n != 1642")
        if (
            patient.get("n_total") != EXPECTED_TEST_PATIENTS
            or patient.get("n_mixed_diagnosis_excluded") != EXPECTED_MIXED_PATIENTS
            or patient.get("n_evaluated", 0)
            + patient.get("n_mixed_diagnosis_excluded", 0)
            != patient.get("n_total")
        ):
            errors.append("headline patient count closure 1649=1642+7 failed")
        for level_name, level, ci_key, majority_ci_key in (
            (
                "biopsy",
                biopsy,
                "patient_cluster_bootstrap_ci",
                "majority_vote_patient_cluster_bootstrap_ci",
            ),
            (
                "patient",
                patient,
                "patient_bootstrap_ci",
                "majority_vote_patient_bootstrap_ci",
            ),
        ):
            for mode, current_ci_key, metrics_key in (
                ("mean", ci_key, "mean_probability_metrics"),
                ("majority", majority_ci_key, "majority_vote_metrics"),
            ):
                ci_rows = level.get(current_ci_key, {})
                if set(ci_rows) != {"accuracy", "macro_f1"}:
                    errors.append(f"{level_name}/{mode} aggregation CI set mismatch")
                for metric in ("accuracy", "macro_f1"):
                    errors.extend(
                        _ci_errors(
                            f"headline/{level_name}/{mode}/{metric}",
                            ci_rows.get(metric),
                            BOOTSTRAP_EXPECTED,
                            level.get(metrics_key, {}).get(metric),
                        )
                    )
        patient_level = analysis.get("patient_level", {})
        aggregation_ci = patient_level.get("aggregation_ci", {})
        aggregation_metric_keys = {
            "biopsy_mean": "biopsy_mean_prob_aggregation",
            "biopsy_majority": "biopsy_majority_vote_aggregation",
            "patient_mean": "mean_prob_aggregation",
            "patient_majority": "majority_vote_aggregation",
        }
        if set(aggregation_ci) != set(aggregation_metric_keys):
            errors.append("patient_level aggregation_ci set mismatch")
        for name, metrics_key in aggregation_metric_keys.items():
            for metric in ("accuracy", "macro_f1"):
                errors.extend(
                    _ci_errors(
                        f"patient_level/{name}/{metric}",
                        aggregation_ci.get(name, {}).get(metric),
                        BOOTSTRAP_EXPECTED,
                        patient_level.get(metrics_key, {}).get(metric),
                    )
                )
        sweep8 = analysis.get("tta_sweep", {}).get("8", {}).get("ms_per_image_ensemble")
        cost8 = analysis.get("inference_cost", {}).get("ensemble_tta8_ms_per_image")
        try:
            if not np.isclose(float(sweep8), float(cost8), rtol=1e-12, atol=1e-9):
                errors.append("TTA8 sweep timing != inference_cost measured timing")
        except (TypeError, ValueError):
            errors.append("TTA8 measured timing missing/non-numeric")
    extended_path = p.out / "extended_ablation_analysis.json"
    if extended_path.is_file():
        extended = _json(extended_path)
        required_arms = {
            "ce_plain",
            "ce_sampler",
            "focal_only",
            "focal_sampler",
            "full",
            "full_cbreweight",
            "cb_only",
            "mixup_only",
            "cutmix_only",
        }
        if set(extended.get("arms", {})) != required_arms:
            errors.append("extended ablation sweep is not complete 9-arm set")
        else:
            required_arm_fields = {
                "flags",
                "metrics",
                "ece",
                "nll",
                "brier_multiclass",
                "minority_overcompensation_index",
                "pred_vs_true_share",
                "evaluation_tta",
            }
            for arm, row in extended["arms"].items():
                if not required_arm_fields.issubset(row):
                    errors.append(f"ablation schema missing core fields: {arm}")
                    continue
                if "ece" not in row and "calibration" not in row:
                    errors.append(f"ablation schema missing ECE: {arm}")
                if row.get("evaluation_tta") != 1:
                    errors.append(f"ablation evaluation TTA is not 1: {arm}")
    checks.append(
        {
            "name": "protocol_and_all_artifacts",
            "ok": not errors,
            "expected_steps_before_validation": len(prerequisite_steps),
            "artifact_hash_count": len(hashes),
        }
    )
    # 单一真值：报告绑定每个前置步骤marker所声明的完整主产物集合。
    hashes = {}
    for step in prerequisite_steps:
        hashes.update(_current_artifact_hashes(step.name))
    marker_hashes = {
        f"results/strict_oof_v1/markers/{step.name}.done": file_sha256(
            p.markers / f"{step.name}.done"
        )
        for step in prerequisite_steps
        if (p.markers / f"{step.name}.done").is_file()
    }
    return {
        "status": "complete",
        "ok": not errors,
        "contract_sha256": contract["contract_sha256"],
        "errors": errors,
        "checks": checks,
        "artifact_sha256": hashes,
        "marker_sha256": marker_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--contract-only", action="store_true")
    group.add_argument("--step")
    group.add_argument("--marker")
    group.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.contract_only:
        errors = validate_contract_only()
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
        return int(bool(errors))
    if args.step:
        errors = validate_step(args.step)
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
        return int(bool(errors))
    if args.marker:
        errors = validate_marker(args.marker, validate_artifact=True)
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
        return int(bool(errors))
    report = validate_full()
    atomic_write_json(paths().validation_report, report)
    print(json.dumps({"ok": report["ok"], "errors": report["errors"]}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
