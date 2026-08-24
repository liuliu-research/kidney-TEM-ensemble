"""严格 Train-OOF 融合、独立 Meta-Cal 选型/标定与终末 Test。"""

from __future__ import annotations

import json
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mutual_info_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import config as legacy
from strict_oof.predict import prediction_paths
from strict_oof.protocol import (
    BACKBONES,
    CLASSES,
    NUM_CLASSES,
    OUTER_FOLDS,
    TRAIN_MODELS,
    atomic_write_json,
    canonical_json_sha256,
    file_sha256,
    load_contract,
    paths,
    patient_of,
    source_rows,
)

BOOTSTRAPS = 1_000


def tta_mean(probabilities: np.ndarray, count: int) -> np.ndarray:
    """对确定 TTA 前缀求平均并重归一化。"""
    result = probabilities[:count].astype(np.float64).mean(axis=0)
    return result / result.sum(axis=1, keepdims=True)


def _load_bundle(path: Path, contract_hash: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["contract_sha256"].item()) != contract_hash:
            raise RuntimeError(f"预测合同不匹配: {path}")
        return {name: archive[name].copy() for name in archive.files}


def _aligned_probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), dtype=np.float64)
    result = np.zeros((len(features), NUM_CLASSES), dtype=np.float64)
    result[:, np.asarray(model.classes_, dtype=int)] = raw
    return result / result.sum(axis=1, keepdims=True)


def _weights(y: np.ndarray, probabilities: dict[str, np.ndarray]) -> np.ndarray:
    scores = np.asarray(
        [
            f1_score(
                y,
                probabilities[model].argmax(1),
                labels=list(range(NUM_CLASSES)),
                average="macro",
                zero_division=0,
            )
            for model in BACKBONES
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(scores)) or scores.sum() <= 0:
        return np.full(len(BACKBONES), 1 / len(BACKBONES))
    return scores / scores.sum()


def _vote(probabilities: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    result = sum(
        weight * probabilities[model] for weight, model in zip(weights, BACKBONES)
    )
    return result / result.sum(axis=1, keepdims=True)


def select_fusion_candidate(
    y_metacal: np.ndarray,
    metacal_candidates: dict[str, np.ndarray],
) -> tuple[str, dict[str, float]]:
    """仅接收独立 Meta-Cal 标签与两个预注册候选。"""
    expected = {"voting_tta8", "stack_lr_tta8"}
    if set(metacal_candidates) != expected:
        raise ValueError(f"融合候选必须严格为 {sorted(expected)}")
    scores = {
        name: float(
            f1_score(
                y_metacal,
                probabilities.argmax(1),
                labels=list(range(NUM_CLASSES)),
                average="macro",
                zero_division=0,
            )
        )
        for name, probabilities in metacal_candidates.items()
    }
    order = ["voting_tta8", "stack_lr_tta8"]
    selected = max(order, key=lambda name: (scores[name], -order.index(name)))
    return selected, scores


def fit_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    return_metadata: bool = False,
) -> float | tuple[float, dict[str, Any]]:
    """仅在 Meta-Cal 上拟合单全局温度。"""
    log_probs = np.log(np.clip(probabilities, 1e-12, 1))

    def objective(temperature: float) -> float:
        logits = log_probs / temperature
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return float(
            -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean()
        )

    bounds = (0.05, 10.0)
    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    temperature = float(result.x)
    if (
        not result.success
        or not np.isfinite(temperature)
        or not np.isfinite(float(result.fun))
        or not bounds[0] <= temperature <= bounds[1]
        or float(result.fun) > objective(1.0) + 1e-10
    ):
        raise RuntimeError(f"temperature optimization failed: {result}")
    metadata = {
        "method": "bounded",
        "bounds": list(bounds),
        "success": bool(result.success),
        "status": int(getattr(result, "status", 0)),
        "nfev": int(result.nfev),
        "objective_nll": float(result.fun),
        "identity_nll": objective(1.0),
    }
    return (temperature, metadata) if return_metadata else temperature


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """应用固定温度。"""
    logits = np.log(np.clip(probabilities, 1e-12, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=1, keepdims=True)


def top_label_ece(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    """计算 top-label ECE。"""
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for index in range(bins):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if mask.any():
            value += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return value


def binary_ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    """计算标准 one-vs-rest 二元 ECE。"""
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for index in range(bins):
        mask = (
            (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
            if index == 0
            else (probabilities > edges[index]) & (probabilities <= edges[index + 1])
        )
        if mask.any():
            value += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probabilities[mask].mean())
            )
    return value


def calibration_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    """返回全局与逐类标定指标。"""
    onehot = np.eye(NUM_CLASSES)[labels]
    per_class = {
        class_name: binary_ece(
            (labels == class_index).astype(float), probabilities[:, class_index]
        )
        for class_index, class_name in enumerate(CLASSES)
    }
    ece = top_label_ece(labels, probabilities)
    balanced_ece = float(np.mean(list(per_class.values())))
    return {
        "top_label_ece": ece,
        # 兼容现有终稿构建器的命名；数值仍是同一 top-label ECE。
        "ece": ece,
        "one_vs_rest_ece": per_class,
        "balanced_one_vs_rest_ece": balanced_ece,
        "classwise_ece": {
            "method": "standard one-vs-rest binary ECE",
            "per_class": per_class,
            "balanced_ece": balanced_ece,
        },
        "brier_multiclass": float(((probabilities - onehot) ** 2).sum(1).mean()),
        "nll": float(
            log_loss(
                labels,
                np.clip(probabilities, 1e-12, 1),
                labels=list(range(NUM_CLASSES)),
            )
        ),
        "reliability": reliability_bins(labels, probabilities),
        "classwise_reliability": classwise_reliability(labels, probabilities),
    }


def reliability_bins(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> list[dict[str, float | int | None]]:
    """返回 top-label 可靠性分箱。"""
    confidence = probabilities.max(1)
    correct = (probabilities.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for index in range(bins):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        rows.append(
            {
                "bin_lo": float(edges[index]),
                "bin_hi": float(edges[index + 1]),
                "n": int(mask.sum()),
                "conf": float(confidence[mask].mean()) if mask.any() else None,
                "acc": float(correct[mask].mean()) if mask.any() else None,
            }
        )
    return rows


def classwise_reliability(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> dict[str, list[dict[str, float | int | None]]]:
    """返回标准 one-vs-rest 逐类可靠性分箱。"""
    edges = np.linspace(0, 1, bins + 1)
    result = {}
    for class_index, class_name in enumerate(CLASSES):
        scores = probabilities[:, class_index]
        binary = (labels == class_index).astype(float)
        rows = []
        for index in range(bins):
            mask = (
                (scores >= edges[index]) & (scores <= edges[index + 1])
                if index == 0
                else (scores > edges[index]) & (scores <= edges[index + 1])
            )
            rows.append(
                {
                    "bin_lo": float(edges[index]),
                    "bin_hi": float(edges[index + 1]),
                    "n": int(mask.sum()),
                    "mean_pred": float(scores[mask].mean()) if mask.any() else None,
                    "observed_freq": float(binary[mask].mean()) if mask.any() else None,
                }
            )
        result[class_name] = rows
    return result


def full_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """分类指标和逐类指标。"""
    predictions = probabilities.argmax(1)
    recalls = recall_score(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
        average=None,
        zero_division=0,
    )
    specificities = []
    for class_index in range(NUM_CLASSES):
        true_positive_class = labels == class_index
        predicted_positive_class = predictions == class_index
        true_negatives = int((~true_positive_class & ~predicted_positive_class).sum())
        false_positives = int((~true_positive_class & predicted_positive_class).sum())
        specificities.append(true_negatives / max(true_negatives + false_positives, 1))
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "cohen_kappa": float(cohen_kappa_score(labels, predictions)),
        "macro_sensitivity": float(np.mean(recalls)),
        "macro_specificity": float(np.mean(specificities)),
        "worst_class_recall": float(np.min(recalls)),
        "auc_ovr_macro": float(
            roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
        ),
        "macro_ap": float(
            np.mean(
                [
                    average_precision_score(
                        (labels == class_index).astype(int),
                        probabilities[:, class_index],
                    )
                    for class_index in range(NUM_CLASSES)
                ]
            )
        ),
    }
    result["per_class"] = {
        class_name: {
            "n": int((labels == class_index).sum()),
            "recall": float(recalls[class_index]),
            "precision": float(
                precision_score(
                    labels,
                    predictions,
                    labels=[class_index],
                    average=None,
                    zero_division=0,
                )[0]
            ),
            "f1": float(
                f1_score(
                    labels,
                    predictions,
                    labels=[class_index],
                    average=None,
                    zero_division=0,
                )[0]
            ),
            "ap": float(
                average_precision_score(
                    (labels == class_index).astype(int),
                    probabilities[:, class_index],
                )
            ),
            "specificity": float(specificities[class_index]),
        }
        for class_index, class_name in enumerate(CLASSES)
    }
    return result


def patient_bootstrap(
    labels: np.ndarray,
    probabilities: np.ndarray,
    patients: np.ndarray,
    functions: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    n_bootstraps: int = BOOTSTRAPS,
) -> dict[str, dict[str, float | int]]:
    """患者级聚类 bootstrap。"""
    groups = np.asarray([patient_of(value) for value in patients])
    unique = np.unique(groups)
    by_patient = {patient: np.where(groups == patient)[0] for patient in unique}
    rng = np.random.default_rng(legacy.SEED)
    values = {name: [] for name in functions}
    point_estimates = {
        name: float(function(labels, probabilities))
        for name, function in functions.items()
    }
    if not all(np.isfinite(value) for value in point_estimates.values()):
        raise RuntimeError("bootstrap point estimate is non-finite")
    attempts = 0
    max_attempts = n_bootstraps * 20
    while len(next(iter(values.values()))) < n_bootstraps:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                "bootstrap could not obtain the requested finite replicates"
            )
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([by_patient[patient] for patient in sampled])
        try:
            draw = {
                name: float(function(labels[indices], probabilities[indices]))
                for name, function in functions.items()
            }
        except ValueError:
            continue
        if not all(np.isfinite(value) for value in draw.values()):
            continue
        for name, value in draw.items():
            values[name].append(value)
    if any(len(current) != n_bootstraps for current in values.values()):
        raise RuntimeError("bootstrap effective count differs from requested count")
    return {
        name: {
            "mean": float(np.mean(current)),
            "point_estimate": point_estimates[name],
            "ci_lo": float(np.percentile(current, 2.5)),
            "ci_hi": float(np.percentile(current, 97.5)),
            "sd": float(np.std(current)),
            "n_boot": len(current),
            "n_boot_requested": n_bootstraps,
            "n_boot_effective": len(current),
            "n_draws_attempted": attempts,
            "bootstrap_unit": "patient cluster",
            "n_observations": int(len(labels)),
            "n_clusters": int(len(unique)),
        }
        for name, current in values.items()
    }


def paired_patient_bootstrap_delta(
    labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    patients: np.ndarray,
    functions: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    n_bootstraps: int = BOOTSTRAPS,
) -> dict[str, dict[str, float | int | bool]]:
    """配对患者聚类 bootstrap，Δ = first - second。"""
    groups = np.asarray([patient_of(value) for value in patients])
    unique = np.unique(groups)
    by_patient = {patient: np.where(groups == patient)[0] for patient in unique}
    rng = np.random.default_rng(legacy.SEED + 17)
    values = {name: [] for name in functions}
    point_estimates = {
        name: float(function(labels, first)) - float(function(labels, second))
        for name, function in functions.items()
    }
    if not all(np.isfinite(value) for value in point_estimates.values()):
        raise RuntimeError("paired bootstrap point estimate is non-finite")
    attempts = 0
    max_attempts = n_bootstraps * 20
    while len(next(iter(values.values()))) < n_bootstraps:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                "paired bootstrap could not obtain requested finite replicates"
            )
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([by_patient[patient] for patient in sampled])
        try:
            draw = {
                name: float(function(labels[indices], first[indices]))
                - float(function(labels[indices], second[indices]))
                for name, function in functions.items()
            }
        except ValueError:
            continue
        if not all(np.isfinite(value) for value in draw.values()):
            continue
        for name, value in draw.items():
            values[name].append(value)
    if any(len(current) != n_bootstraps for current in values.values()):
        raise RuntimeError("paired bootstrap effective count differs from requested")
    result = {}
    for name, current in values.items():
        low = float(np.percentile(current, 2.5))
        high = float(np.percentile(current, 97.5))
        result[name] = {
            "mean": float(np.mean(current)),
            "point_estimate": point_estimates[name],
            "ci_lo": low,
            "ci_hi": high,
            "sd": float(np.std(current)),
            "n_boot": len(current),
            "n_boot_requested": n_bootstraps,
            "n_boot_effective": len(current),
            "n_draws_attempted": attempts,
            "excludes_zero": bool(low > 0 or high < 0),
            "bootstrap_unit": "patient cluster",
            "n_observations": int(len(labels)),
            "n_clusters": int(len(unique)),
        }
    return result


def fit_predicted_class_temperatures(
    probabilities: np.ndarray,
    labels: np.ndarray,
    global_temperature: float,
    min_samples: int = 50,
) -> tuple[dict[int, float], dict[int, int], list[int]]:
    """拟合与应用都按推理时可见的预测类分组。"""
    predicted = probabilities.argmax(1)
    temperatures = {}
    counts = {}
    fallback = []
    for class_index in range(NUM_CLASSES):
        mask = predicted == class_index
        counts[class_index] = int(mask.sum())
        if mask.sum() >= min_samples:
            temperatures[class_index] = fit_temperature(
                probabilities[mask], labels[mask]
            )
        else:
            temperatures[class_index] = global_temperature
            fallback.append(class_index)
    return temperatures, counts, fallback


def _mcnemar(
    labels: np.ndarray, first: np.ndarray, second: np.ndarray
) -> dict[str, float | int]:
    first_correct = first.argmax(1) == labels
    second_correct = second.argmax(1) == labels
    first_only = int((first_correct & ~second_correct).sum())
    second_only = int((~first_correct & second_correct).sum())
    statistic = (abs(first_only - second_only) - 1) ** 2 / max(
        first_only + second_only, 1
    )
    return {
        "final_only": first_only,
        "comparator_only": second_only,
        "b_a_only": first_only,
        "c_b_only": second_only,
        "statistic": float(statistic),
        "p": float(chi2.sf(statistic, 1)),
    }


def _aggregate_by_key(
    labels: np.ndarray,
    probabilities: np.ndarray,
    biopsy_ids: np.ndarray,
    keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """按活检或患者键聚合，返回用于聚类 bootstrap 的患者键。"""
    aggregated_labels = []
    aggregated_probs = []
    aggregated_votes = []
    cluster_patients = []
    mixed = 0
    for key in np.unique(keys):
        indices = np.where(keys == key)[0]
        current_labels = np.unique(labels[indices])
        if len(current_labels) != 1:
            mixed += 1
            continue
        aggregated_labels.append(int(current_labels[0]))
        aggregated_probs.append(probabilities[indices].mean(axis=0))
        votes = np.bincount(
            probabilities[indices].argmax(1), minlength=NUM_CLASSES
        ).astype(np.float64)
        aggregated_votes.append(votes / votes.sum())
        cluster_patients.append(patient_of(str(biopsy_ids[indices[0]])))
    return (
        np.asarray(aggregated_labels, dtype=np.int64),
        np.asarray(aggregated_probs, dtype=np.float64),
        np.asarray(aggregated_votes, dtype=np.float64),
        np.asarray(cluster_patients),
        mixed,
    )


def analysis_input_sha256() -> dict[str, str]:
    """返回分析直接消费的全部预测 bundle 哈希。"""
    root = paths().exp
    result: dict[str, str] = {}
    for model in TRAIN_MODELS:
        for fold in range(OUTER_FOLDS):
            bundle, _shards, _meta, _lock = prediction_paths(
                "outer", model, "train_oof", fold=fold
            )
            result[bundle.resolve().relative_to(root).as_posix()] = file_sha256(bundle)
        for subset in ("metacal", "test"):
            bundle, _shards, _meta, _lock = prediction_paths("terminal", model, subset)
            result[bundle.resolve().relative_to(root).as_posix()] = file_sha256(bundle)
    return result


def _already_complete(
    output: Path, contract_hash: str, input_sha256: dict[str, str]
) -> bool:
    if not output.is_file():
        return False
    try:
        value = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("contract_sha256") == contract_hash
        and value.get("status") == "complete"
        and value.get("input_prediction_sha256") == input_sha256
    )


def main() -> int:  # noqa: PLR0915
    contract = load_contract()
    from strict_oof.preflight import assert_frozen_runtime

    assert_frozen_runtime(contract)
    p = paths()
    output = p.out / "analysis.json"
    input_prediction_sha256 = analysis_input_sha256()
    if _already_complete(output, contract["contract_sha256"], input_prediction_sha256):
        print("strict analysis complete, skip")
        return 0
    train_rows, metacal_rows, test_rows, _hashes = source_rows()
    y_train = np.asarray([legacy.CLASS_TO_IDX[row["cls"]] for row in train_rows])
    y_metacal = np.asarray([legacy.CLASS_TO_IDX[row["cls"]] for row in metacal_rows])
    y_test = np.asarray([legacy.CLASS_TO_IDX[row["cls"]] for row in test_rows])
    patient_train = np.asarray([row["pt"] for row in train_rows])
    patient_metacal = np.asarray([row["pt"] for row in metacal_rows])
    patient_test = np.asarray([row["pt"] for row in test_rows])

    train_oof_raw: dict[str, np.ndarray] = {}
    for model in TRAIN_MODELS:
        assembled = np.full((8, len(train_rows), NUM_CLASSES), np.nan, dtype=np.float32)
        seen = np.zeros(len(train_rows), dtype=np.int8)
        for fold in range(OUTER_FOLDS):
            bundle, _shards, _meta, _lock = prediction_paths(
                "outer", model, "train_oof", fold=fold
            )
            archive = _load_bundle(bundle, contract["contract_sha256"])
            indices = archive["global_indices"].astype(int)
            if np.any(seen[indices]):
                raise RuntimeError(f"{model} Train OOF 重复覆盖")
            if not np.array_equal(archive["labels"], y_train[indices]):
                raise RuntimeError(f"{model} Train OOF 标签/顺序不一致")
            assembled[:, indices] = archive["probs"]
            seen[indices] = 1
        if np.any(seen != 1) or np.any(~np.isfinite(assembled)):
            raise RuntimeError(f"{model} Train OOF 未完整覆盖")
        train_oof_raw[model] = assembled

    terminal_raw: dict[str, dict[str, np.ndarray]] = {"metacal": {}, "test": {}}
    terminal_timing: dict[str, dict[str, np.ndarray]] = {
        "metacal": {},
        "test": {},
    }
    for subset, expected_y in (("metacal", y_metacal), ("test", y_test)):
        for model in TRAIN_MODELS:
            bundle, _shards, _meta, _lock = prediction_paths("terminal", model, subset)
            archive = _load_bundle(bundle, contract["contract_sha256"])
            if not np.array_equal(archive["labels"], expected_y):
                raise RuntimeError(f"{model}/{subset} 标签顺序不一致")
            terminal_raw[subset][model] = archive["probs"]
            terminal_timing[subset][model] = archive["tta_seconds"].astype(np.float64)

    train_tta8 = {model: tta_mean(train_oof_raw[model], 8) for model in BACKBONES}
    weights = _weights(y_train, train_tta8)
    train_features = np.hstack([train_tta8[model] for model in BACKBONES])
    with warnings.catch_warnings(record=True) as stack_warnings:
        warnings.simplefilter("always")
        stack = make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2_000)
        ).fit(train_features, y_train)
    stack_lr = stack.steps[-1][1]
    stack_n_iter = int(np.max(np.asarray(stack_lr.n_iter_)))
    if (
        any(issubclass(item.category, ConvergenceWarning) for item in stack_warnings)
        or stack_n_iter >= int(stack_lr.max_iter)
        or not np.isfinite(stack_lr.coef_).all()
        or not np.isfinite(stack_lr.intercept_).all()
    ):
        raise RuntimeError("primary LR stacker did not converge cleanly")
    stack_parameter_hash = canonical_json_sha256(
        {
            "coef": stack_lr.coef_.tolist(),
            "intercept": stack_lr.intercept_.tolist(),
            "classes": stack_lr.classes_.tolist(),
        }
    )
    primary_stack_fit = {
        "fit_partition": "complete original-Train OOF TTA8",
        "solver": str(stack_lr.solver),
        "C": float(stack_lr.C),
        "n_iter": stack_n_iter,
        "max_iter": int(stack_lr.max_iter),
        "converged": True,
        "feature_dim": int(train_features.shape[1]),
        "parameter_sha256": stack_parameter_hash,
        "warnings": [str(item.message) for item in stack_warnings],
    }

    # R3.3 非线性元学习器只是描述性敏感性分析。三个模型均只在
    # Train OOF TTA1 特征上拟合；MLP/GBDT 不进入 Meta-Cal 选型候选集。
    train_tta1 = {model: tta_mean(train_oof_raw[model], 1) for model in BACKBONES}
    train_features_tta1 = np.hstack([train_tta1[model] for model in BACKBONES])
    sensitivity_estimators = {
        "stack_lr_tta1": LogisticRegression(C=1.0, max_iter=2_000),
        "stack_mlp_tta1": MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=800, random_state=contract["seed"]
        ),
        "stack_gbdt_tta1": HistGradientBoostingClassifier(
            random_state=contract["seed"]
        ),
    }
    sensitivity_models: dict[str, Any] = {}
    sensitivity_fit: dict[str, Any] = {}
    for name, estimator in sensitivity_estimators.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = make_pipeline(StandardScaler(), estimator).fit(
                train_features_tta1, y_train
            )
        final_estimator = fitted.steps[-1][1]
        n_iter_raw = getattr(final_estimator, "n_iter_", None)
        n_iter = int(np.max(np.asarray(n_iter_raw))) if n_iter_raw is not None else None
        max_iter = getattr(final_estimator, "max_iter", None)
        sensitivity_models[name] = fitted
        sensitivity_fit[name] = {
            "fit_partition": "complete original-Train OOF TTA1",
            "used_for_fusion_selection": False,
            "random_state": getattr(final_estimator, "random_state", None),
            "n_iter": n_iter,
            "max_iter": int(max_iter) if max_iter is not None else None,
            "converged": (
                bool(n_iter < int(max_iter))
                if n_iter is not None and max_iter is not None
                else None
            ),
            "warnings": [str(item.message) for item in caught],
        }

    metacal_tta8 = {
        model: tta_mean(terminal_raw["metacal"][model], 8) for model in BACKBONES
    }
    test_tta8 = {model: tta_mean(terminal_raw["test"][model], 8) for model in BACKBONES}
    metacal_candidates = {
        "voting_tta8": _vote(metacal_tta8, weights),
        "stack_lr_tta8": _aligned_probabilities(
            stack,
            np.hstack([metacal_tta8[model] for model in BACKBONES]),
        ),
    }
    test_candidates = {
        "voting_tta8": _vote(test_tta8, weights),
        "stack_lr_tta8": _aligned_probabilities(
            stack, np.hstack([test_tta8[model] for model in BACKBONES])
        ),
    }
    metacal_features_tta1 = np.hstack(
        [tta_mean(terminal_raw["metacal"][model], 1) for model in BACKBONES]
    )
    test_features_tta1 = np.hstack(
        [tta_mean(terminal_raw["test"][model], 1) for model in BACKBONES]
    )
    sensitivity_metacal = {
        name: _aligned_probabilities(model, metacal_features_tta1)
        for name, model in sensitivity_models.items()
    }
    sensitivity_test = {
        name: _aligned_probabilities(model, test_features_tta1)
        for name, model in sensitivity_models.items()
    }
    selected, selection_scores = select_fusion_candidate(y_metacal, metacal_candidates)
    p_metacal = metacal_candidates[selected]
    p_test = test_candidates[selected]
    temperature, temperature_fit = fit_temperature(
        p_metacal, y_metacal, return_metadata=True
    )
    p_metacal_calibrated = apply_temperature(p_metacal, temperature)
    p_test_calibrated = apply_temperature(p_test, temperature)
    class_temperatures, class_temperature_counts, fallback_classes = (
        fit_predicted_class_temperatures(
            p_metacal, y_metacal, global_temperature=temperature
        )
    )
    p_test_class_temperature = p_test.copy()
    predicted_test = p_test.argmax(1)
    for class_index in range(NUM_CLASSES):
        mask = predicted_test == class_index
        if mask.any():
            p_test_class_temperature[mask] = apply_temperature(
                p_test[mask], class_temperatures[class_index]
            )

    referral_rows: dict[str, Any] = {}
    selected_threshold = None
    for threshold in legacy.REFERRAL_THRESHOLDS:
        metacal_auto = p_metacal_calibrated.max(1) >= threshold
        test_auto = p_test_calibrated.max(1) >= threshold
        meta_accuracy = (
            float(
                (
                    p_metacal_calibrated.argmax(1)[metacal_auto]
                    == y_metacal[metacal_auto]
                ).mean()
            )
            if metacal_auto.any()
            else None
        )
        if (
            selected_threshold is None
            and meta_accuracy is not None
            and meta_accuracy >= 0.90
        ):
            selected_threshold = threshold
        referral_rows[str(threshold)] = {
            "metacal_n_total": int(len(y_metacal)),
            "metacal_n_autonomous": int(metacal_auto.sum()),
            "metacal_n_referred": int((~metacal_auto).sum()),
            "metacal_coverage": float(metacal_auto.mean()),
            "metacal_autonomous_accuracy": meta_accuracy,
            "metacal_auto_acc": meta_accuracy,
            "metacal_referred_class_dist": {
                class_name: int(((y_metacal == class_index) & ~metacal_auto).sum())
                for class_index, class_name in enumerate(CLASSES)
            },
            "metacal_per_class_referral_rate": {
                class_name: float((~metacal_auto)[y_metacal == class_index].mean())
                for class_index, class_name in enumerate(CLASSES)
            },
            "test_n_total": int(len(y_test)),
            "test_n_autonomous": int(test_auto.sum()),
            "test_n_referred": int((~test_auto).sum()),
            "test_coverage": float(test_auto.mean()),
            "coverage": float(test_auto.mean()),
            "test_autonomous_accuracy": (
                float(
                    (p_test_calibrated.argmax(1)[test_auto] == y_test[test_auto]).mean()
                )
                if test_auto.any()
                else None
            ),
            "auto_accuracy": (
                float(
                    (p_test_calibrated.argmax(1)[test_auto] == y_test[test_auto]).mean()
                )
                if test_auto.any()
                else None
            ),
            "test_referred_accuracy": (
                float(
                    (
                        p_test_calibrated.argmax(1)[~test_auto] == y_test[~test_auto]
                    ).mean()
                )
                if (~test_auto).any()
                else None
            ),
            "referred_accuracy": (
                float(
                    (
                        p_test_calibrated.argmax(1)[~test_auto] == y_test[~test_auto]
                    ).mean()
                )
                if (~test_auto).any()
                else None
            ),
            "test_referred_class_dist": {
                class_name: int(((y_test == class_index) & ~test_auto).sum())
                for class_index, class_name in enumerate(CLASSES)
            },
            "referred_class_dist": {
                class_name: int(((y_test == class_index) & ~test_auto).sum())
                for class_index, class_name in enumerate(CLASSES)
            },
            "test_per_class_referral_rate": {
                class_name: float((~test_auto)[y_test == class_index].mean())
                for class_index, class_name in enumerate(CLASSES)
            },
            "per_class_referral_rate": {
                class_name: float((~test_auto)[y_test == class_index].mean())
                for class_index, class_name in enumerate(CLASSES)
            },
        }

    metacal_single_scores = {
        model: float(
            f1_score(
                y_metacal,
                metacal_tta8[model].argmax(1),
                average="macro",
                zero_division=0,
            )
        )
        for model in BACKBONES
    }
    train_oof_single_scores = {
        model: float(
            f1_score(
                y_train,
                train_tta1[model].argmax(1),
                labels=list(range(NUM_CLASSES)),
                average="macro",
                zero_division=0,
            )
        )
        for model in BACKBONES
    }
    best_single = max(BACKBONES, key=lambda model: train_oof_single_scores[model])
    # 稿件中的融合增益是预注册的复合对比：final TTA8 对
    # 完整 Train OOF TTA1 预选单骨干的 terminal TTA1。
    best_single_test = tta_mean(terminal_raw["test"][best_single], 1)
    resnet_metacal = tta_mean(terminal_raw["metacal"]["resnet50"], 8)
    resnet_test = tta_mean(terminal_raw["test"]["resnet50"], 8)

    bootstrap_functions = {
        "accuracy": lambda y, prob: accuracy_score(y, prob.argmax(1)),
        "macro_f1": lambda y, prob: f1_score(
            y, prob.argmax(1), average="macro", zero_division=0
        ),
        "balanced_accuracy": lambda y, prob: balanced_accuracy_score(y, prob.argmax(1)),
        "mcc": lambda y, prob: matthews_corrcoef(y, prob.argmax(1)),
        "cohen_kappa": lambda y, prob: cohen_kappa_score(y, prob.argmax(1)),
        "auc_ovr_macro": lambda y, prob: roc_auc_score(
            y, prob, multi_class="ovr", average="macro"
        ),
        "macro_sensitivity": lambda y, prob: float(
            recall_score(
                y,
                prob.argmax(1),
                labels=list(range(NUM_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "macro_specificity": lambda y, prob: full_metrics(y, prob)["macro_specificity"],
        "macro_ap": lambda y, prob: float(
            np.mean(
                [
                    average_precision_score(
                        (y == class_index).astype(int), prob[:, class_index]
                    )
                    for class_index in range(NUM_CLASSES)
                ]
            )
        ),
        "worst_class_recall": lambda y, prob: float(
            np.min(
                recall_score(
                    y,
                    prob.argmax(1),
                    labels=list(range(NUM_CLASSES)),
                    average=None,
                    zero_division=0,
                )
            )
        ),
    }
    calibration_bootstrap_functions = {
        # 终稿表格的 ECE 即 top-label ECE；键名与消费者一致。
        "ece": top_label_ece,
        "brier_multiclass": lambda y, prob: float(
            ((prob - np.eye(NUM_CLASSES)[y]) ** 2).sum(1).mean()
        ),
        "nll": lambda y, prob: float(
            log_loss(
                y,
                np.clip(prob, 1e-12, 1),
                labels=list(range(NUM_CLASSES)),
            )
        ),
    }
    paired_functions = {
        "accuracy": lambda y, prob: accuracy_score(y, prob.argmax(1)),
        "macro_f1": lambda y, prob: f1_score(
            y, prob.argmax(1), average="macro", zero_division=0
        ),
        "balanced_accuracy": lambda y, prob: balanced_accuracy_score(y, prob.argmax(1)),
        "macro_ap": lambda y, prob: float(
            np.mean(
                [
                    average_precision_score(
                        (y == class_index).astype(int), prob[:, class_index]
                    )
                    for class_index in range(NUM_CLASSES)
                ]
            )
        ),
        "worst_class_recall": lambda y, prob: float(
            np.min(
                recall_score(
                    y,
                    prob.argmax(1),
                    labels=list(range(NUM_CLASSES)),
                    average=None,
                    zero_division=0,
                )
            )
        ),
    }
    for class_index, class_name in enumerate(CLASSES):
        paired_functions[f"recall_{class_name}"] = (
            lambda y, prob, class_index=class_index: float(
                recall_score(
                    y,
                    prob.argmax(1),
                    labels=[class_index],
                    average=None,
                    zero_division=0,
                )[0]
            )
        )
    per_class_ci = {}
    for class_index, class_name in enumerate(CLASSES):
        functions = {
            "recall": lambda y, prob, class_index=class_index: float(
                recall_score(
                    y,
                    prob.argmax(1),
                    labels=[class_index],
                    average=None,
                    zero_division=0,
                )[0]
            ),
            "precision": lambda y, prob, class_index=class_index: float(
                precision_score(
                    y,
                    prob.argmax(1),
                    labels=[class_index],
                    average=None,
                    zero_division=0,
                )[0]
            ),
            "f1": lambda y, prob, class_index=class_index: float(
                f1_score(
                    y,
                    prob.argmax(1),
                    labels=[class_index],
                    average=None,
                    zero_division=0,
                )[0]
            ),
            "ap": lambda y, prob, class_index=class_index: float(
                average_precision_score(
                    (y == class_index).astype(int), prob[:, class_index]
                )
            ),
        }
        per_class_ci[class_name] = patient_bootstrap(
            y_test, p_test, patient_test, functions, n_bootstraps=500
        )
    aggregation_bootstrap_functions = {
        "accuracy": lambda y, prob: accuracy_score(y, prob.argmax(1)),
        "macro_f1": lambda y, prob: f1_score(
            y, prob.argmax(1), average="macro", zero_division=0
        ),
    }
    biopsy_keys = patient_test.astype(str)
    patient_keys = np.asarray([patient_of(value) for value in patient_test])
    biopsy_y, biopsy_p, biopsy_vote, biopsy_clusters, mixed_biopsies = (
        _aggregate_by_key(y_test, p_test_calibrated, patient_test, biopsy_keys)
    )
    patient_y, patient_p, patient_vote, patient_clusters, mixed_patients = (
        _aggregate_by_key(y_test, p_test_calibrated, patient_test, patient_keys)
    )
    diversity = {
        "prob_correlation": {},
        "disagreement": {},
        "kappa": {},
        "mutual_info": {},
        "both_wrong_rate": {},
    }
    single_tta1 = {
        model: tta_mean(terminal_raw["test"][model], 1) for model in BACKBONES
    }
    for first_model, second_model in combinations(BACKBONES, 2):
        key = f"{first_model}|{second_model}"
        first_prob = single_tta1[first_model]
        second_prob = single_tta1[second_model]
        first_pred = first_prob.argmax(1)
        second_pred = second_prob.argmax(1)
        diversity["prob_correlation"][key] = float(
            np.corrcoef(first_prob.ravel(), second_prob.ravel())[0, 1]
        )
        diversity["disagreement"][key] = float((first_pred != second_pred).mean())
        diversity["kappa"][key] = float(cohen_kappa_score(first_pred, second_pred))
        diversity["mutual_info"][key] = float(
            mutual_info_score(first_pred, second_pred)
        )
        diversity["both_wrong_rate"][key] = float(
            ((first_pred != y_test) & (second_pred != y_test)).mean()
        )
    pr_curves = {}
    for class_index, class_name in enumerate(CLASSES):
        precision_values, recall_values, threshold_values = precision_recall_curve(
            (y_test == class_index).astype(int), p_test[:, class_index]
        )
        pr_curves[class_name] = {
            "precision": precision_values.round(8).tolist(),
            "recall": recall_values.round(8).tolist(),
            "thresholds": threshold_values.round(8).tolist(),
        }
    dca = {}
    for class_index, class_name in enumerate(CLASSES):
        binary = (y_test == class_index).astype(int)
        prevalence = float(binary.mean())
        scores = p_test_calibrated[:, class_index]
        rows = []
        for threshold in np.linspace(0.05, 0.60, 23):
            positive = scores >= threshold
            true_positive = float((positive & (binary == 1)).mean())
            false_positive = float((positive & (binary == 0)).mean())
            rows.append(
                {
                    "pt": round(float(threshold), 3),
                    "nb_model": true_positive
                    - false_positive * threshold / (1 - threshold),
                    "nb_treat_all": prevalence
                    - (1 - prevalence) * threshold / (1 - threshold),
                    "nb_treat_none": 0.0,
                }
            )
        dca[class_name] = {"prevalence": prevalence, "curve": rows}
    timing_per_model = {
        model: float(terminal_timing["test"][model][0]) * 1_000 / len(y_test)
        for model in TRAIN_MODELS
    }
    ensemble_tta8_ms = (
        sum(float(terminal_timing["test"][model][:8].sum()) for model in BACKBONES)
        * 1_000
        / len(y_test)
    )
    final_ci = patient_bootstrap(y_test, p_test, patient_test, bootstrap_functions)
    calibration_ci = {
        "before": patient_bootstrap(
            y_test,
            p_test,
            patient_test,
            calibration_bootstrap_functions,
        ),
        "after": patient_bootstrap(
            y_test,
            p_test_calibrated,
            patient_test,
            calibration_bootstrap_functions,
        ),
    }
    delta_vs_best = paired_patient_bootstrap_delta(
        y_test,
        p_test,
        best_single_test,
        patient_test,
        paired_functions,
    )
    delta_vs_resnet = paired_patient_bootstrap_delta(
        y_test,
        p_test,
        resnet_test,
        patient_test,
        paired_functions,
    )
    confusion = confusion_matrix(
        y_test, p_test.argmax(1), labels=list(range(NUM_CLASSES))
    )
    row_denominators = confusion.sum(1, keepdims=True)
    row_normalized = np.divide(
        confusion,
        row_denominators,
        out=np.zeros_like(confusion, dtype=float),
        where=row_denominators > 0,
    )
    ensembles = {
        "voting_tta8": full_metrics(y_test, test_candidates["voting_tta8"]),
        "stack_lr_tta8": full_metrics(y_test, test_candidates["stack_lr_tta8"]),
        "voting_tta1": full_metrics(
            y_test,
            _vote(
                {
                    model: tta_mean(terminal_raw["test"][model], 1)
                    for model in BACKBONES
                },
                weights,
            ),
        ),
        "voting_uniform_tta1": full_metrics(
            y_test,
            _vote(
                {
                    model: tta_mean(terminal_raw["test"][model], 1)
                    for model in BACKBONES
                },
                np.full(len(BACKBONES), 1 / len(BACKBONES)),
            ),
        ),
        "voting_uniform_tta8": full_metrics(
            y_test,
            _vote(
                test_tta8,
                np.full(len(BACKBONES), 1 / len(BACKBONES)),
            ),
        ),
        **{
            name: full_metrics(y_test, probabilities)
            for name, probabilities in sensitivity_test.items()
        },
    }
    fusion_protocol = {
        "protocol": (
            "backbone-level patient-grouped five-fold Train OOF; fusion "
            "fit on complete Train OOF; independent Meta-Cal selection"
        ),
        "splitter": "StratifiedGroupKFold",
        "n_splits": OUTER_FOLDS,
        "seed": contract["seed"],
        "folds": contract["fold_records"],
        "group_definition": (
            "patient code before the first underscore; repeat biopsies "
            "remain in one fold"
        ),
        "oof_selection_candidates": ["voting_tta8", "stack_lr_tta8"],
        "candidate_roles": {
            name: {
                "used_for_fusion_selection": True,
                "fit_partition": "complete original-Train OOF",
                "selection_partition": "independent Meta-Cal",
            }
            for name in ("voting_tta8", "stack_lr_tta8")
        },
        "post_hoc_terminal_test_sensitivity": [
            "stack_mlp_tta1",
            "stack_gbdt_tta1",
        ],
        "candidate_metacal_scores": selection_scores,
        "selected_candidate": selected,
        "selection_partition": "independent Meta-Cal",
        "test_labels_used_for_selection": False,
        "resnet50_excluded_from_ensemble": True,
        "sensitivity_roles": {
            name: {
                "used_for_fusion_selection": False,
                "fit_partition": "complete original-Train OOF TTA1",
                "evaluation_partitions": ["independent Meta-Cal", "terminal Test"],
                "interpretation": "post-hoc descriptive sensitivity only",
            }
            for name in ("stack_mlp_tta1", "stack_gbdt_tta1")
        },
    }
    tta_sensitivity: dict[str, Any] = {}
    for count in (1, 2, 4, 8, 16):
        current_mc = {
            model: tta_mean(terminal_raw["metacal"][model], count)
            for model in BACKBONES
        }
        current_test = {
            model: tta_mean(terminal_raw["test"][model], count) for model in BACKBONES
        }
        stack_fit_depth = count if count <= 8 else 8
        if count <= 8:
            current_train = {
                model: tta_mean(train_oof_raw[model], count) for model in BACKBONES
            }
            current_stack = make_pipeline(
                StandardScaler(), LogisticRegression(C=1.0, max_iter=2_000)
            ).fit(
                np.hstack([current_train[model] for model in BACKBONES]),
                y_train,
            )
        else:
            # Train OOF 只预注册了 TTA8。TTA16 因此使用已在
            # Train-OOF TTA8 拟合的冻结 LR，仅作cross-depth描述性敏感性。
            current_stack = stack
        stack_metacal_metrics = full_metrics(
            y_metacal,
            _aligned_probabilities(
                current_stack,
                np.hstack([current_mc[model] for model in BACKBONES]),
            ),
        )
        stack_test_metrics = full_metrics(
            y_test,
            _aligned_probabilities(
                current_stack,
                np.hstack([current_test[model] for model in BACKBONES]),
            ),
        )
        measured_ms = float(
            sum(
                float(terminal_timing["test"][model][:count].sum())
                for model in BACKBONES
            )
            * 1_000
            / len(y_test)
        )
        tta_sensitivity[str(count)] = {
            "voting_metacal": full_metrics(y_metacal, _vote(current_mc, weights)),
            "voting_test": full_metrics(y_test, _vote(current_test, weights)),
            "stack_lr_metacal": stack_metacal_metrics,
            "stack_lr_test": stack_test_metrics,
            "stack_lr_fit_tta_depth": stack_fit_depth,
            "stack_lr_cross_depth_application": count == 16,
            "selection_role": (
                "prespecified_primary" if count == 8 else "sensitivity_only"
            ),
            "ms_per_image_ensemble": measured_ms,
        }

    result = {
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "input_prediction_sha256": input_prediction_sha256,
        "singles": {
            model: {
                "tta1": full_metrics(y_test, tta_mean(terminal_raw["test"][model], 1)),
                "tta8": full_metrics(y_test, tta_mean(terminal_raw["test"][model], 8)),
                "ms_per_image": timing_per_model[model],
            }
            for model in TRAIN_MODELS
        },
        "voting_weights": {
            model: float(weight) for model, weight in zip(BACKBONES, weights)
        },
        "tta_sweep": {
            count: {
                "voting": row["voting_test"],
                "stack_lr": row["stack_lr_test"],
                "selection_role": row["selection_role"],
                "ms_per_image_ensemble": row["ms_per_image_ensemble"],
            }
            for count, row in tta_sensitivity.items()
        },
        "ensembles": ensembles,
        "final_key": selected,
        "selection_on_metacal": {
            name: {
                "macro_f1": score,
                "estimate": "independent_metacal",
            }
            for name, score in selection_scores.items()
        },
        "fusion_selection_protocol": fusion_protocol,
        "primary_lr_stacker_fit": primary_stack_fit,
        "nonlinear_stacker_sensitivity": {
            "fit": sensitivity_fit,
            "metacal": {
                name: full_metrics(y_metacal, probabilities)
                for name, probabilities in sensitivity_metacal.items()
            },
            "test": {
                name: full_metrics(y_test, probabilities)
                for name, probabilities in sensitivity_test.items()
            },
            "used_for_fusion_selection": False,
            "test_labels_used_for_fitting_or_selection": False,
        },
        "protocol": {
            "backbone_oof_fit_partition": "original Train",
            "backbone_outer_folds": OUTER_FOLDS,
            "backbone_outer_heldout_labels_used_for_training_or_epoch_selection": False,
            "fusion_fit_partition": "complete original-Train OOF",
            "fusion_selection_partition": "independent original Meta-Cal",
            "temperature_and_referral_fit_partition": "independent original Meta-Cal",
            "terminal_evaluation_partition": "original Test",
            "test_labels_used_for_any_selection": False,
        },
        "train_oof_standalone": {
            model: full_metrics(y_train, tta_mean(train_oof_raw[model], 8))
            for model in TRAIN_MODELS
        },
        "voting_weights_from_train_oof": {
            model: float(weight) for model, weight in zip(BACKBONES, weights)
        },
        "fusion_selection": {
            "candidates": ["voting_tta8", "stack_lr_tta8"],
            "metacal_macro_f1": selection_scores,
            "selected": selected,
            "tie_breaker": "voting precedes LR stacking",
            "test_labels_used": False,
        },
        "metacal_selected_uncalibrated": full_metrics(y_metacal, p_metacal),
        "terminal_test_uncalibrated": full_metrics(y_test, p_test),
        "terminal_test_calibrated_classification": full_metrics(
            y_test, p_test_calibrated
        ),
        "calibration": {
            "temperature": temperature,
            "temperature_fit": temperature_fit,
            "fit_partition": "Meta-Cal selected-candidate probabilities",
            "fit_source": "independent Meta-Cal selected-candidate probabilities",
            "metacal_before": calibration_metrics(y_metacal, p_metacal),
            "metacal_after": calibration_metrics(y_metacal, p_metacal_calibrated),
            "test_before": calibration_metrics(y_test, p_test),
            "test_after": calibration_metrics(y_test, p_test_calibrated),
            "before": calibration_metrics(y_test, p_test),
            "after": calibration_metrics(y_test, p_test_calibrated),
        },
        "classwise_temperature_probe": {
            "per_class_T": {
                CLASSES[class_index]: float(value)
                for class_index, value in class_temperatures.items()
            },
            "fit_partition": "independent Meta-Cal",
            "fit_group_definition": "predicted class",
            "application_group_definition": "predicted class",
            "n_fit_images_by_predicted_class": {
                CLASSES[class_index]: class_temperature_counts[class_index]
                for class_index in range(NUM_CLASSES)
            },
            "fallback_to_global_temperature": [
                CLASSES[class_index] for class_index in fallback_classes
            ],
            "balanced_ece_global_T": calibration_metrics(y_test, p_test_calibrated)[
                "balanced_one_vs_rest_ece"
            ],
            "balanced_ece_classwise_T": calibration_metrics(
                y_test, p_test_class_temperature
            )["balanced_one_vs_rest_ece"],
            "note": (
                "Sensitivity only; fitting and application both group by "
                "predicted class. The primary system uses one global temperature."
            ),
        },
        "referral": {
            "selection_partition": "Meta-Cal",
            "metacal_probability_source": (
                "temperature-scaled independent Meta-Cal probabilities "
                "from the selected fusion candidate"
            ),
            "selection_rule": (
                "lowest threshold with Meta-Cal autonomous accuracy >= 0.90"
            ),
            "chosen_threshold": selected_threshold,
            "metacal_class_totals": {
                class_name: int((y_metacal == class_index).sum())
                for class_index, class_name in enumerate(CLASSES)
            },
            "test_class_totals": {
                class_name: int((y_test == class_index).sum())
                for class_index, class_name in enumerate(CLASSES)
            },
            "thresholds": referral_rows,
        },
        "tta_sensitivity": tta_sensitivity,
        "best_single_selection": {
            "selected": best_single,
            "candidate_scores": train_oof_single_scores,
            "train_oof_tta1_macro_f1": train_oof_single_scores,
            "metacal_macro_f1_descriptive": metacal_single_scores,
            "selection_source": "complete original-Train OOF TTA1",
            "test_labels_used": False,
            "test_labels_used_for_selection": False,
            "metacal_labels_used": False,
            "metacal_labels_used_for_selection": False,
            "criterion": "highest complete Train OOF macro-F1 at TTA1",
            "terminal_comparison_depth": "TTA1",
        },
        "same_protocol_resnet50": {
            "metacal": full_metrics(y_metacal, resnet_metacal),
            "test": full_metrics(y_test, resnet_test),
            "excluded_from_ensemble": True,
        },
        "mcnemar": {
            "final_vs_best_single": _mcnemar(y_test, p_test, best_single_test),
            "final_vs_resnet50": _mcnemar(y_test, p_test, resnet_test),
        },
        "mcnemar_final_vs_best_single": {
            "best_single": best_single,
            "selection_source": "complete original-Train OOF macro-F1 at TTA1",
            "comparison": "final selected fusion TTA8 versus selected single TTA1",
            **_mcnemar(y_test, p_test, best_single_test),
        },
        "test_patient_bootstrap_ci": final_ci,
        "final_ci": final_ci,
        "calibration_ci": calibration_ci,
        "per_class_ci": per_class_ci,
        "delta_vs_best_single_ci": {
            "best_single": best_single,
            "note": (
                "Delta = final system - best single backbone selected on "
                "complete Train OOF TTA1, paired "
                "patient-cluster bootstrap"
            ),
            "comparison": "final selected fusion TTA8 versus selected single TTA1",
            "deltas": delta_vs_best,
        },
        "delta_vs_resnet50_ci": {
            "comparator": "same-protocol ResNet50",
            "deltas": delta_vs_resnet,
        },
        "diversity": diversity,
        "confusion_matrix": {
            "labels": list(CLASSES),
            "counts": confusion.tolist(),
            "row_normalized": row_normalized.round(6).tolist(),
        },
        "pr_curves": pr_curves,
        "headline_aggregation": {
            "biopsy": {
                "n_total": int(len(np.unique(biopsy_keys))),
                "n_evaluated": int(len(biopsy_y)),
                "n_mixed_diagnosis_excluded": mixed_biopsies,
                "mean_probability_metrics": full_metrics(biopsy_y, biopsy_p),
                "majority_vote_metrics": full_metrics(biopsy_y, biopsy_vote),
                "patient_cluster_bootstrap_ci": patient_bootstrap(
                    biopsy_y,
                    biopsy_p,
                    biopsy_clusters,
                    aggregation_bootstrap_functions,
                ),
                "majority_vote_patient_cluster_bootstrap_ci": patient_bootstrap(
                    biopsy_y,
                    biopsy_vote,
                    biopsy_clusters,
                    aggregation_bootstrap_functions,
                ),
                "bootstrap_unit": (
                    "patient; all repeat biopsies from a sampled patient "
                    "are resampled together"
                ),
                "mean_probability_source": "globally temperature-calibrated Test probabilities",
            },
            "patient": {
                "n_total": int(len(np.unique(patient_keys))),
                "n_evaluated": int(len(patient_y)),
                "n_mixed_diagnosis_excluded": mixed_patients,
                "mean_probability_metrics": full_metrics(patient_y, patient_p),
                "majority_vote_metrics": full_metrics(patient_y, patient_vote),
                "patient_bootstrap_ci": patient_bootstrap(
                    patient_y,
                    patient_p,
                    patient_clusters,
                    aggregation_bootstrap_functions,
                ),
                "majority_vote_patient_bootstrap_ci": patient_bootstrap(
                    patient_y,
                    patient_vote,
                    patient_clusters,
                    aggregation_bootstrap_functions,
                ),
                "bootstrap_unit": "patient",
                "mean_probability_source": "globally temperature-calibrated Test probabilities",
            },
        },
        "patient_level": {
            "n_biopsies_test": int(len(np.unique(biopsy_keys))),
            "n_patients_test": int(len(np.unique(patient_keys))),
            "n_patients_evaluated": int(len(patient_y)),
            "n_patients_mixed_diagnoses_excluded": mixed_patients,
            "biopsy_mean_prob_aggregation": full_metrics(biopsy_y, biopsy_p),
            "biopsy_majority_vote_aggregation": full_metrics(biopsy_y, biopsy_vote),
            "mean_prob_aggregation": full_metrics(patient_y, patient_p),
            "majority_vote_aggregation": full_metrics(patient_y, patient_vote),
            "aggregation_ci": {
                "biopsy_mean": patient_bootstrap(
                    biopsy_y,
                    biopsy_p,
                    biopsy_clusters,
                    aggregation_bootstrap_functions,
                ),
                "biopsy_majority": patient_bootstrap(
                    biopsy_y,
                    biopsy_vote,
                    biopsy_clusters,
                    aggregation_bootstrap_functions,
                ),
                "patient_mean": patient_bootstrap(
                    patient_y,
                    patient_p,
                    patient_clusters,
                    aggregation_bootstrap_functions,
                ),
                "patient_majority": patient_bootstrap(
                    patient_y,
                    patient_vote,
                    patient_clusters,
                    aggregation_bootstrap_functions,
                ),
            },
        },
        "dca": dca,
        "inference_cost": {
            "per_backbone_ms_per_image": timing_per_model,
            "ensemble_tta1_ms_per_image": float(
                sum(timing_per_model[model] for model in BACKBONES)
            ),
            "ensemble_tta8_ms_per_image": float(ensemble_tta8_ms),
            "measurement": (
                "sum of measured terminal Test forward-pass times; "
                "JPEG loading is excluded"
            ),
        },
        "sample_counts": {
            "train_images": len(y_train),
            "metacal_images": len(y_metacal),
            "test_images": len(y_test),
            "train_patients": len({patient_of(value) for value in patient_train}),
            "metacal_patients": len({patient_of(value) for value in patient_metacal}),
            "test_patients": len({patient_of(value) for value in patient_test}),
        },
    }
    atomic_write_json(output, result)
    print(f"selected={selected} temperature={temperature:.6f}")
    print(json.dumps(result["terminal_test_uncalibrated"], indent=2)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
