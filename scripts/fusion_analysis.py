"""决策融合 + 审稿意见全家桶分析（全部基于真实预测）.

前置: 5 骨干 + resnet50 的 metacal/test 预测 npz 已生成。
输出: results/analysis.json（论文全部表格/图的数据源）,
      results/analysis_tables.csv 系列。

覆盖: 软投票/LR·MLP·GBDT stacking/TTA 扫描/温度标定/ECE·逐类ECE·balanced
ECE·Brier·NLL/可靠性图/13→11 类全指标 + Balanced Acc·MCC·κ·宏敏感/特异
/PR-AUC/逐类 CI(患者级 bootstrap)/多样性(相关·不一致·κ·互信息)/转诊分析
/McNemar/混淆矩阵/传统 ML 基线(冻结特征)。
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score,
    log_loss, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C

MODELS = list(C.BACKBONES)          # 5 骨干
BOOT_N = 1000
RNG = np.random.default_rng(C.SEED)
OOF_SELECTION_CANDIDATES = ("voting_tta8", "stack_lr_tta8")
POST_HOC_TERMINAL_TEST_SENSITIVITY = (
    "stack_mlp_tta1",
    "stack_gbdt_tta1",
)


# ---------------- 载入 ----------------
def load_preds(tag: str, subset: str):
    z = np.load(C.PRED_DIR / f"{tag}__{subset}.npz", allow_pickle=False)
    return (z["probs"].astype(np.float32), z["labels"], z["patients"],
            float(z["ms_per_image_single"]))


def tta_mean(probs: np.ndarray, n: int) -> np.ndarray:
    p = probs[:n].mean(0)
    return p / p.sum(1, keepdims=True)      # float16 存储误差归一化


def patient_group_ids(patient_or_biopsy_ids: np.ndarray) -> np.ndarray:
    """从活检编号中提取患者分组键。

    预测文件中的活检编号以患者代码为第一个下划线前缀；
    重复活检必须始终位于同一折。
    """
    return np.asarray([str(value).split("_", 1)[0]
                       for value in patient_or_biopsy_ids])


def patient_grouped_oof_splits(
        y: np.ndarray,
        patient_or_biopsy_ids: np.ndarray,
        n_splits: int = C.FUSION_OOF_FOLDS,
        seed: int = C.SEED,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], list[dict]]:
    """生成固定 seed 的分层患者分组 OOF 折，并对泄漏失败关闭。"""
    y = np.asarray(y)
    groups = patient_group_ids(np.asarray(patient_or_biopsy_ids))
    if y.ndim != 1 or len(y) != len(groups):
        raise ValueError("y 与 patient_or_biopsy_ids 必须是等长一维数组")
    if n_splits < 2:
        raise ValueError("OOF 折数必须至少为 2")
    if len(np.unique(groups)) < n_splits:
        raise ValueError("患者数少于 OOF 折数")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    records: list[dict] = []
    assignment = np.full(len(y), -1, dtype=np.int16)
    all_classes = set(np.unique(y).tolist())
    for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(np.zeros(len(y)), y, groups)):
        train_groups = set(groups[train_idx].tolist())
        valid_groups = set(groups[valid_idx].tolist())
        if train_groups & valid_groups:
            raise RuntimeError(f"OOF 第 {fold} 折存在患者泄漏")
        if np.any(assignment[valid_idx] != -1):
            raise RuntimeError(f"OOF 第 {fold} 折重复覆盖了验证样本")
        train_classes = set(np.unique(y[train_idx]).tolist())
        if train_classes != all_classes:
            missing = sorted(all_classes - train_classes)
            raise RuntimeError(f"OOF 第 {fold} 折训练集缺少类别: {missing}")
        assignment[valid_idx] = fold
        splits.append((train_idx, valid_idx))
        records.append({
            "fold": int(fold),
            "n_train_images": int(len(train_idx)),
            "n_validation_images": int(len(valid_idx)),
            "n_train_patient_groups": int(len(train_groups)),
            "n_validation_patient_groups": int(len(valid_groups)),
            "patient_group_overlap": 0,
        })
    if np.any(assignment < 0):
        raise RuntimeError("OOF 折未覆盖全部 meta-cal 样本")
    return groups, splits, records


def _aligned_predict_proba(model, x: np.ndarray,
                           n_classes: int) -> np.ndarray:
    """将 sklearn 模型概率列对齐到固定类别编号。"""
    raw = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=int)
    if np.any(classes < 0) or np.any(classes >= n_classes):
        raise RuntimeError("元学习器返回了越界类别编号")
    probs = np.zeros((len(x), n_classes), dtype=np.float64)
    probs[:, classes] = raw
    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0):
        raise RuntimeError("元学习器返回了非法概率")
    return probs / row_sums


def grouped_oof_predict_proba(
        x: np.ndarray,
        y: np.ndarray,
        splits: list[tuple[np.ndarray, np.ndarray]],
        estimator,
        n_classes: int,
) -> np.ndarray:
    """在每折内独立拟合标准化与元学习器，返回 OOF 概率。"""
    x = np.asarray(x)
    y = np.asarray(y)
    probs = np.full((len(y), n_classes), np.nan, dtype=np.float64)
    for train_idx, valid_idx in splits:
        model = make_pipeline(StandardScaler(), clone(estimator))
        model.fit(x[train_idx], y[train_idx])
        probs[valid_idx] = _aligned_predict_proba(
            model, x[valid_idx], n_classes)
    if np.any(~np.isfinite(probs)):
        raise RuntimeError("OOF 概率未被完整填充")
    return probs


def _macro_f1_weights(
        y: np.ndarray,
        probs_by_model: dict[str, np.ndarray],
        model_names: list[str],
        indices: np.ndarray | None = None,
) -> np.ndarray:
    """仅使用指定样本的标签估计软投票权重。"""
    if indices is None:
        indices = np.arange(len(y))
    scores = np.asarray([
        f1_score(
            y[indices],
            probs_by_model[name][indices].argmax(1),
            labels=list(range(probs_by_model[name].shape[1])),
            average="macro",
            zero_division=0,
        )
        for name in model_names
    ], dtype=np.float64)
    if not np.all(np.isfinite(scores)) or scores.sum() <= 0:
        return np.full(len(model_names), 1.0 / len(model_names))
    return scores / scores.sum()


def _weighted_vote(
        probs_by_model: dict[str, np.ndarray],
        model_names: list[str],
        weights: np.ndarray,
        indices: np.ndarray | None = None,
) -> np.ndarray:
    """按给定权重合并模型概率。"""
    if indices is None:
        indices = np.arange(len(next(iter(probs_by_model.values()))))
    combined = sum(
        weight * probs_by_model[name][indices]
        for weight, name in zip(weights, model_names)
    )
    return combined / combined.sum(axis=1, keepdims=True)


def grouped_oof_weighted_vote(
        weight_fit_probs: dict[str, np.ndarray],
        vote_probs: dict[str, np.ndarray],
        y: np.ndarray,
        splits: list[tuple[np.ndarray, np.ndarray]],
        model_names: list[str],
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """每折仅用训练患者估计权重，再预测该折留出患者。"""
    n_classes = next(iter(vote_probs.values())).shape[1]
    oof = np.full((len(y), n_classes), np.nan, dtype=np.float64)
    fold_weights: list[dict[str, float]] = []
    for train_idx, valid_idx in splits:
        weights = _macro_f1_weights(
            y, weight_fit_probs, model_names, indices=train_idx)
        oof[valid_idx] = _weighted_vote(
            vote_probs, model_names, weights, indices=valid_idx)
        fold_weights.append({
            name: float(weight)
            for name, weight in zip(model_names, weights)
        })
    if np.any(~np.isfinite(oof)):
        raise RuntimeError("加权投票 OOF 概率未被完整填充")
    return oof, fold_weights


def select_candidate_from_oof(
        y: np.ndarray,
        candidate_oof_probs: dict[str, np.ndarray],
) -> tuple[str, dict[str, dict[str, float | str]]]:
    """仅根据 meta-cal OOF macro-F1 选择最终融合候选。"""
    if not candidate_oof_probs:
        raise ValueError("至少需要一个融合候选")
    scores: dict[str, dict[str, float | str]] = {}
    for name, probs in candidate_oof_probs.items():
        scores[name] = {
            "macro_f1": float(f1_score(
                y,
                probs.argmax(1),
                labels=list(range(probs.shape[1])),
                average="macro",
                zero_division=0,
            )),
            "estimate": "patient_grouped_oof",
        }
    selected = max(scores, key=lambda name: scores[name]["macro_f1"])
    return selected, scores


def fusion_selection_role_metadata() -> dict:
    """明示区分 OOF 选型候选与终末 test 事后敏感性对照。"""
    roles = {
        name: {
            "selection_role": "patient_grouped_oof_selection_candidate",
            "used_for_fusion_selection": True,
            "has_grouped_oof_selection_score": True,
        }
        for name in OOF_SELECTION_CANDIDATES
    }
    roles.update({
        name: {
            "selection_role": "post_hoc_terminal_test_sensitivity",
            "used_for_fusion_selection": False,
            "has_grouped_oof_selection_score": False,
            "fit_partition": "full_metacal",
            "evaluation_partition": "terminal_test",
        }
        for name in POST_HOC_TERMINAL_TEST_SENSITIVITY
    })
    return {
        "oof_selection_candidates": list(OOF_SELECTION_CANDIDATES),
        "post_hoc_terminal_test_sensitivity": list(
            POST_HOC_TERMINAL_TEST_SENSITIVITY),
        "candidate_roles": roles,
    }


def select_best_single_from_metacal(
        y: np.ndarray,
        metacal_probs: dict[str, np.ndarray],
) -> tuple[str, dict[str, float]]:
    """仅用 meta-cal macro-F1 预选 McNemar/bootstrap 的单骨干对照。"""
    scores = {
        name: float(f1_score(
            y,
            probs.argmax(1),
            labels=list(range(probs.shape[1])),
            average="macro",
            zero_division=0,
        ))
        for name, probs in metacal_probs.items()
    }
    selected = max(scores, key=scores.get)
    return selected, scores


# ---------------- 指标 ----------------
def full_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    yp = p.argmax(1)
    n_cls = p.shape[1]
    out = {
        "accuracy": accuracy_score(y, yp),
        "macro_f1": f1_score(y, yp, average="macro"),
        "weighted_f1": f1_score(y, yp, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(y, yp),
        "mcc": matthews_corrcoef(y, yp),
        "cohen_kappa": cohen_kappa_score(y, yp),
        "macro_sensitivity": recall_score(y, yp, average="macro"),
        "auc_ovr_macro": roc_auc_score(y, p, multi_class="ovr",
                                       average="macro"),
        "worst_class_recall": float(min(
            recall_score(y, yp, average=None, zero_division=0))),
    }
    spec, aps = [], []
    for c in range(n_cls):
        yb = (y == c).astype(int)
        pb = (yp == c).astype(int)
        tn = int(((yb == 0) & (pb == 0)).sum())
        fp = int(((yb == 0) & (pb == 1)).sum())
        spec.append(tn / max(tn + fp, 1))
        aps.append(average_precision_score(yb, p[:, c]))
    out["macro_specificity"] = float(np.mean(spec))
    out["macro_ap"] = float(np.mean(aps))
    out["per_class"] = {
        C.CLASSES[c]: {
            "recall": float(recall_score(y, yp, labels=[c], average=None,
                                         zero_division=0)[0]),
            "precision": float(precision_score(y, yp, labels=[c],
                                               average=None,
                                               zero_division=0)[0]),
            "f1": float(f1_score(y, yp, labels=[c], average=None,
                                 zero_division=0)[0]),
            "ap": float(aps[c]),
            "specificity": float(spec[c]),
            "n": int((y == c).sum()),
        } for c in range(n_cls)
    }
    return out


def macro_specificity(y, p) -> float:
    """宏平均特异度（一对多）——供 bootstrap CI 复用，口径同 full_metrics."""
    yp = p.argmax(1)
    spec = []
    for c in range(p.shape[1]):
        yb = y == c
        pb = yp == c
        tn = int((~yb & ~pb).sum())
        fp = int((~yb & pb).sum())
        spec.append(tn / max(tn + fp, 1))
    return float(np.mean(spec))


def macro_ap(y, p) -> float:
    """宏平均 average precision（PR-AUC）——供 bootstrap CI 复用."""
    return float(np.mean([
        average_precision_score((y == c).astype(int), p[:, c])
        for c in range(p.shape[1])]))


def patient_bootstrap_ci(y, p, pts, metric_fns: dict, n_boot=BOOT_N):
    """患者级 bootstrap 95% CI（按人聚类：PT 前缀，重复活检并入同一人）."""
    persons = np.array([str(x).split("_")[0] for x in pts])
    uniq = np.unique(persons)
    idx_by_pt = {u: np.where(persons == u)[0] for u in uniq}
    stats = {k: [] for k in metric_fns}
    for _ in range(n_boot):
        sel = RNG.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by_pt[u] for u in sel])
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < len(C.CLASSES):
            # 罕见类缺席的重采样轮次跳过 AUC 类指标
            pass
        for k, fn in metric_fns.items():
            try:
                stats[k].append(fn(yy, pp))
            except ValueError:
                continue
    return {k: {"mean": float(np.mean(v)),
                "ci_lo": float(np.percentile(v, 2.5)),
                "ci_hi": float(np.percentile(v, 97.5)),
                "sd": float(np.std(v)), "n_boot": len(v)}
            for k, v in stats.items()}


def paired_bootstrap_delta(y, p_a, p_b, pts, metric_fns: dict,
                           n_boot=BOOT_N):
    """配对患者级 bootstrap：同一轮重采样同时评估两套系统，返回 Δ = A − B.

    回应 R5.7 / R5.2——审稿人要的是"这一具体增益是否可靠"，只给最终系统
    自身的 CI 不构成回答；配对重采样消除了两次独立 bootstrap 之间的
    重采样噪声，Δ 的百分位区间可直接判读是否跨零。
    """
    persons = np.array([str(x).split("_")[0] for x in pts])
    uniq = np.unique(persons)
    idx_by_pt = {u: np.where(persons == u)[0] for u in uniq}
    stats = {k: [] for k in metric_fns}
    for _ in range(n_boot):
        sel = RNG.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by_pt[u] for u in sel])
        yy, pa, pb = y[idx], p_a[idx], p_b[idx]
        for k, fn in metric_fns.items():
            try:
                stats[k].append(float(fn(yy, pa)) - float(fn(yy, pb)))
            except ValueError:
                continue
    out = {}
    for k, v in stats.items():
        lo = float(np.percentile(v, 2.5))
        hi = float(np.percentile(v, 97.5))
        out[k] = {"mean": float(np.mean(v)), "ci_lo": lo, "ci_hi": hi,
                  "sd": float(np.std(v)), "n_boot": len(v),
                  "excludes_zero": bool(lo > 0 or hi < 0)}
    return out


# ---------------- 校准 ----------------
def fit_temperature(logp: np.ndarray, y: np.ndarray) -> float:
    def nll(t):
        z = logp / t
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    logp = np.log(np.clip(probs, 1e-12, 1))
    z = logp / t
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def fit_predicted_class_temperatures(
        probs: np.ndarray,
        y: np.ndarray,
        global_temperature: float,
        n_classes: int,
        min_samples: int = 50,
) -> tuple[dict[int, float], dict[int, int], list[int]]:
    """按预测类别拟合逐类温度，并对小样本组回退全局温度。

    拟合和应用必须使用同一个、推理时可获得的分组变量。这里二者都使用
    argmax 预测类别；不得用真实类别拟合后再按预测类别应用。
    """
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y)
    if probs.ndim != 2 or len(probs) != len(y):
        raise ValueError("probs 与 y 必须具有相同样本数")
    if probs.shape[1] != n_classes:
        raise ValueError("probs 的类别数与 n_classes 不一致")
    if min_samples < 1:
        raise ValueError("min_samples 必须为正整数")

    predicted = probs.argmax(1)
    logp = np.log(np.clip(probs, 1e-12, 1))
    temperatures: dict[int, float] = {}
    counts: dict[int, int] = {}
    fallback_classes: list[int] = []
    for class_index in range(n_classes):
        mask = predicted == class_index
        count = int(mask.sum())
        counts[class_index] = count
        if count >= min_samples:
            temperatures[class_index] = fit_temperature(logp[mask], y[mask])
        else:
            temperatures[class_index] = float(global_temperature)
            fallback_classes.append(class_index)
    return temperatures, counts, fallback_classes


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def binary_probability_ece(
        target: np.ndarray,
        probability: np.ndarray,
        bins: int = 15,
) -> float:
    """计算一个类别的一对其余 expected calibration error。"""
    target = np.asarray(target, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    if target.shape != probability.shape or target.ndim != 1:
        raise ValueError("target 与 probability 必须是等长一维数组")
    edges = np.linspace(0, 1, bins + 1)
    score = 0.0
    for index in range(bins):
        if index == 0:
            mask = (probability >= edges[index]) & (
                probability <= edges[index + 1])
        else:
            mask = (probability > edges[index]) & (
                probability <= edges[index + 1])
        if mask.any():
            score += mask.mean() * abs(
                target[mask].mean() - probability[mask].mean())
    return float(score)


def classwise_ece(y, p, bins=15) -> dict:
    """计算标准 one-vs-rest 逐类 ECE 及其不加权均值。"""
    out = {
        C.CLASSES[class_index]: binary_probability_ece(
            (y == class_index).astype(float),
            p[:, class_index],
            bins,
        )
        for class_index in range(p.shape[1])
    }
    out["balanced_ece"] = float(np.mean(list(out.values())))
    return out


def reliability_bins(y, p, bins=15):
    conf = p.max(1)
    corr = (p.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        rows.append({"bin_lo": edges[i], "bin_hi": edges[i + 1],
                     "n": int(m.sum()),
                     "conf": float(conf[m].mean()) if m.any() else None,
                     "acc": float(corr[m].mean()) if m.any() else None})
    return rows


def classwise_reliability(y, p, bins=10) -> dict:
    """逐类可靠性曲线（一对多）——R5.5 明确要求 class-wise reliability
    diagrams；逐类 ECE 条形图回答的是另一个问题，不能替代。

    对每个类 c：把该类的预测概率 p[:, c] 分箱，箱内报告平均预测概率与
    实际为 c 的比例，即标准的一对多校准曲线。
    """
    edges = np.linspace(0, 1, bins + 1)
    out = {}
    for c in range(p.shape[1]):
        pc = p[:, c]
        yb = (y == c).astype(float)
        rows = []
        for i in range(bins):
            lo, hi = edges[i], edges[i + 1]
            m = (pc > lo) & (pc <= hi) if i else (pc >= lo) & (pc <= hi)
            rows.append({
                "bin_lo": float(lo), "bin_hi": float(hi),
                "n": int(m.sum()),
                "mean_pred": float(pc[m].mean()) if m.any() else None,
                "observed_freq": float(yb[m].mean()) if m.any() else None,
            })
        out[C.CLASSES[c]] = rows
    return out


def calib_suite(y, p) -> dict:
    onehot = np.eye(p.shape[1])[y]
    return {
        "ece": ece(y, p),
        "classwise_ece": classwise_ece(y, p),
        "brier_multiclass": float(((p - onehot) ** 2).sum(1).mean()),
        "brier_top1": float(brier_score_loss(
            (p.argmax(1) == y).astype(int), p.max(1), pos_label=1)),
        "nll": float(log_loss(y, np.clip(p, 1e-12, 1),
                              labels=list(range(p.shape[1])))),
        "reliability": reliability_bins(y, p),
        "classwise_reliability": classwise_reliability(y, p),
    }


# ---------------- McNemar ----------------
def mcnemar(y, pa, pb) -> dict:
    ca, cb = pa.argmax(1) == y, pb.argmax(1) == y
    b = int((ca & ~cb).sum())
    c = int((~ca & cb).sum())
    stat = (abs(b - c) - 1) ** 2 / max(b + c, 1)
    return {"b_a_only": b, "c_b_only": c, "stat": stat,
            "p": float(chi2.sf(stat, 1))}


def main() -> None:  # noqa: PLR0915
    # ---------- 载入全部预测 ----------
    mc, te = {}, {}
    timing = {}
    for m in MODELS + ["resnet50"]:
        try:
            mc[m] = load_preds(m, "metacal")
            te[m] = load_preds(m, "test")
            timing[m] = te[m][3]
        except FileNotFoundError:
            print(f"WARN missing preds for {m}")
    y_mc, pts_mc = mc[MODELS[0]][1], mc[MODELS[0]][2]
    y_te, pts_te = te[MODELS[0]][1], te[MODELS[0]][2]
    A = {}          # 结果容器

    # ---------- 单模型（TTA=1 与 TTA=8） ----------
    A["singles"] = {}
    for m in MODELS + ["resnet50"]:
        if m not in te:
            continue
        p1 = tta_mean(te[m][0], 1)
        p8 = tta_mean(te[m][0], 8)
        A["singles"][m] = {"tta1": full_metrics(y_te, p1),
                           "tta8": full_metrics(y_te, p8),
                           "ms_per_image": timing[m]}

    # ---------- 软投票（Meta-Cal 定权：macro-F1 归一） ----------
    mc_tta1 = {m: tta_mean(mc[m][0], 1) for m in MODELS}
    w = _macro_f1_weights(y_mc, mc_tta1, MODELS)
    A["voting_weights"] = dict(zip(MODELS, w.round(4).tolist()))

    def vote(preds: dict, n_tta: int):
        probs = {m: tta_mean(preds[m][0], n_tta) for m in MODELS}
        return _weighted_vote(probs, MODELS, w)

    # ---------- Stacking（Meta-Cal 训练；特征=5×11 softmax） ----------
    def stack_feats(preds: dict, n_tta: int):
        return np.hstack([tta_mean(preds[m][0], n_tta) for m in MODELS])

    groups_mc, oof_splits, oof_fold_records = patient_grouped_oof_splits(
        y_mc,
        pts_mc,
        n_splits=C.FUSION_OOF_FOLDS,
        seed=C.SEED,
    )
    Xmc, Xte = stack_feats(mc, 1), stack_feats(te, 1)
    meta_models = {
        "stack_lr": LogisticRegression(C=1.0, max_iter=2000),
        "stack_mlp": MLPClassifier(hidden_layer_sizes=(64,), max_iter=800,
                                   random_state=C.SEED),
        "stack_gbdt": HistGradientBoostingClassifier(random_state=C.SEED),
    }
    stack_te_probs = {}
    for name, mm in meta_models.items():
        fitted = make_pipeline(StandardScaler(), clone(mm)).fit(Xmc, y_mc)
        stack_te_probs[name] = _aligned_predict_proba(
            fitted, Xte, C.NUM_CLASSES)

    # ---------- TTA 扫描（投票 与 LR stacking） ----------
    A["tta_sweep"] = {}
    for n in C.TTA_SWEEP:
        pv = vote(te, n)
        Xte_n = stack_feats(te, n)
        Xmc_n = stack_feats(mc, n)
        lr_n = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000),
        ).fit(Xmc_n, y_mc)
        ps = _aligned_predict_proba(lr_n, Xte_n, C.NUM_CLASSES)
        ms1 = float(np.mean([timing[m] for m in MODELS]))
        A["tta_sweep"][str(n)] = {
            "voting": full_metrics(y_te, pv),
            "stack_lr": full_metrics(y_te, ps),
            "ms_per_image_ensemble": round(ms1 * len(MODELS) * n, 2),
        }

    # ---------- 主配置选择（患者分组 OOF，不在拟合集自评） ----------
    mc_tta8 = {m: tta_mean(mc[m][0], 8) for m in MODELS}
    voting_oof, voting_fold_weights = grouped_oof_weighted_vote(
        mc_tta1,
        mc_tta8,
        y_mc,
        oof_splits,
        MODELS,
    )
    Xmc8 = stack_feats(mc, 8)
    lr8_estimator = LogisticRegression(C=1.0, max_iter=2000)
    stack_lr_oof = grouped_oof_predict_proba(
        Xmc8,
        y_mc,
        oof_splits,
        lr8_estimator,
        C.NUM_CLASSES,
    )
    candidate_oof_probs = {
        "voting_tta8": voting_oof,
        "stack_lr_tta8": stack_lr_oof,
    }
    final_key, selection_scores = select_candidate_from_oof(
        y_mc, candidate_oof_probs)
    role_metadata = fusion_selection_role_metadata()
    if list(selection_scores) != role_metadata["oof_selection_candidates"]:
        raise RuntimeError("OOF 候选分数与预注册选型候选不一致")
    A["selection_on_metacal"] = selection_scores
    A["fusion_selection_protocol"] = {
        "protocol": (
            "fixed patient-grouped stratified out-of-fold comparison "
            "within the meta-calibration set"
        ),
        "splitter": "StratifiedGroupKFold",
        "n_splits": C.FUSION_OOF_FOLDS,
        "seed": C.SEED,
        "group_definition": (
            "patient code before the first underscore; repeat biopsies "
            "from one patient remain in one fold"
        ),
        "n_images": int(len(y_mc)),
        "n_patient_groups": int(len(np.unique(groups_mc))),
        "folds": oof_fold_records,
        "weighted_vote_fold_weights": voting_fold_weights,
        "candidate_oof_scores": selection_scores,
        **role_metadata,
        "primary_metric": "macro_f1",
        "tie_breaker": (
            "candidate declaration order; weighted voting precedes stacking"
        ),
        "selected_candidate": final_key,
        "selection_reason": (
            f"highest patient-grouped OOF macro-F1 on meta-calibration: "
            f"{final_key}={selection_scores[final_key]['macro_f1']:.6f}"
        ),
        "test_labels_used_for_selection": False,
        "final_fit": (
            "after selection, fusion parameters are refit on all "
            "meta-calibration samples and used only to predict test"
        ),
    }

    # 选型后才在全部 meta-cal 上拟合最终元学习器，仅用于 test 预测。
    lr8 = make_pipeline(StandardScaler(), clone(lr8_estimator)).fit(
        Xmc8, y_mc)
    Xte8 = stack_feats(te, 8)

    def vote_uniform(preds: dict, n_tta: int):
        p = sum(tta_mean(preds[m][0], n_tta) for m in MODELS) / len(MODELS)
        return p / p.sum(1, keepdims=True)

    final_probs = {
        "voting_tta8": vote(te, 8),
        "stack_lr_tta8": _aligned_predict_proba(
            lr8, Xte8, C.NUM_CLASSES),
        "stack_mlp_tta1": stack_te_probs["stack_mlp"],
        "stack_gbdt_tta1": stack_te_probs["stack_gbdt"],
        "stack_lr_tta1": stack_te_probs["stack_lr"],
        "voting_tta1": vote(te, 1),
        "voting_uniform_tta1": vote_uniform(te, 1),   # 常规等权集成基线(R3.2)
        "voting_uniform_tta8": vote_uniform(te, 8),
    }
    A["ensembles"] = {k: full_metrics(y_te, v)
                      for k, v in final_probs.items()}

    # ---------- 最终系统 = Meta-Cal 患者分组 OOF macro-F1 最优者 ----------
    A["final_key"] = final_key
    p_final_te = final_probs[final_key]

    # ---------- 温度标定（最终候选的 Meta-Cal OOF 概率拟合） ----------
    p_final_mc = candidate_oof_probs[final_key]
    logp_mc = np.log(np.clip(p_final_mc, 1e-12, 1))
    T = fit_temperature(logp_mc, y_mc)
    p_cal_te = apply_temperature(p_final_te, T)
    A["calibration"] = {
        "temperature": T,
        "fit_source": "selected candidate's patient-grouped meta-cal OOF probabilities",
        "before": calib_suite(y_te, p_final_te),
        "after": calib_suite(y_te, p_cal_te),
    }
    # 逐类温度对照（R5.4 保险分析：全局单温 vs 每个预测类一温）
    per_cls_T_idx, per_cls_fit_counts, fallback_classes = (
        fit_predicted_class_temperatures(
            p_final_mc,
            y_mc,
            global_temperature=T,
            n_classes=C.NUM_CLASSES,
        )
    )
    per_cls_T = {
        C.CLASSES[class_index]: temperature
        for class_index, temperature in per_cls_T_idx.items()
    }
    p_pct = p_final_te.copy()
    yp_arg = p_final_te.argmax(1)
    for c in range(C.NUM_CLASSES):
        m = yp_arg == c
        if m.any():
            p_pct[m] = apply_temperature(p_final_te[m],
                                         per_cls_T[C.CLASSES[c]])
    A["classwise_temperature_probe"] = {
        "per_class_T": {k: round(v, 4) for k, v in per_cls_T.items()},
        "fit_partition": "patient-grouped meta-cal OOF probabilities",
        "fit_group_definition": "predicted class",
        "application_group_definition": "predicted class",
        "n_fit_images_by_predicted_class": {
            C.CLASSES[class_index]: per_cls_fit_counts[class_index]
            for class_index in range(C.NUM_CLASSES)
        },
        "fallback_to_global_temperature": [
            C.CLASSES[class_index] for class_index in fallback_classes
        ],
        "balanced_ece_global_T": A["calibration"]["after"][
            "classwise_ece"]["balanced_ece"],
        "balanced_ece_classwise_T": classwise_ece(y_te, p_pct)[
            "balanced_ece"],
        "note": (
            "保险对照：正文默认全局单温；逐类温度的拟合与应用均按预测类分组"
        ),
    }

    # ---------- 95% CI（患者级 bootstrap） ----------
    fns = {
        "accuracy": lambda yy, pp: accuracy_score(yy, pp.argmax(1)),
        "macro_f1": lambda yy, pp: f1_score(yy, pp.argmax(1),
                                            average="macro"),
        "balanced_accuracy": lambda yy, pp: balanced_accuracy_score(
            yy, pp.argmax(1)),
        "mcc": lambda yy, pp: matthews_corrcoef(yy, pp.argmax(1)),
        "cohen_kappa": lambda yy, pp: cohen_kappa_score(yy, pp.argmax(1)),
        "auc_ovr_macro": lambda yy, pp: roc_auc_score(
            yy, pp, multi_class="ovr", average="macro"),
        # R4.5 要求"所有主要指标"都带 CI——宏特异度与宏 AP 原先缺失，
        # 导致 Table 2 该两格只能填"—"、正文 macro AP 裸奔
        "macro_specificity": lambda yy, pp: macro_specificity(yy, pp),
        "macro_sensitivity": lambda yy, pp: recall_score(
            yy, pp.argmax(1), average="macro"),
        "macro_ap": lambda yy, pp: macro_ap(yy, pp),
        "worst_class_recall": lambda yy, pp: float(min(
            recall_score(yy, pp.argmax(1), average=None, zero_division=0))),
    }
    A["final_ci"] = patient_bootstrap_ci(y_te, p_final_te, pts_te, fns)
    # 校准指标（ECE/Brier/NLL）的患者级 bootstrap CI（回应 R5.5：
    # 审稿人点名要校准指标的置信区间，不能用分类指标 CI 顶替）
    calib_fns = {
        "ece": lambda yy, pp: ece(yy, pp),
        "brier_multiclass": lambda yy, pp: float(
            ((pp - np.eye(pp.shape[1])[yy]) ** 2).sum(1).mean()),
        "nll": lambda yy, pp: float(log_loss(
            yy, np.clip(pp, 1e-12, 1), labels=list(range(pp.shape[1])))),
    }
    A["calibration_ci"] = {
        "before": patient_bootstrap_ci(y_te, p_final_te, pts_te, calib_fns),
        "after": patient_bootstrap_ci(y_te, p_cal_te, pts_te, calib_fns),
    }
    # 逐类 recall/precision/F1/AP 的 CI
    per_cls_ci = {}
    for c, cls in enumerate(C.CLASSES):
        fns_c = {
            "recall": lambda yy, pp, c=c: recall_score(
                yy, pp.argmax(1), labels=[c], average=None,
                zero_division=0)[0],
            "precision": lambda yy, pp, c=c: precision_score(
                yy, pp.argmax(1), labels=[c], average=None,
                zero_division=0)[0],
            "f1": lambda yy, pp, c=c: f1_score(
                yy, pp.argmax(1), labels=[c], average=None,
                zero_division=0)[0],
            "ap": lambda yy, pp, c=c: average_precision_score(
                (yy == c).astype(int), pp[:, c]),
        }
        per_cls_ci[cls] = patient_bootstrap_ci(y_te, p_final_te, pts_te,
                                               fns_c, n_boot=500)
    A["per_class_ci"] = per_cls_ci

    # ---------- 多样性 ----------
    argmaxes = {m: tta_mean(te[m][0], 1).argmax(1) for m in MODELS}
    flat = {m: tta_mean(te[m][0], 1).ravel() for m in MODELS}
    corr, disag, kap, mi = {}, {}, {}, {}
    from sklearn.metrics import mutual_info_score
    for a, b in combinations(MODELS, 2):
        key = f"{a}|{b}"
        corr[key] = float(np.corrcoef(flat[a], flat[b])[0, 1])
        disag[key] = float((argmaxes[a] != argmaxes[b]).mean())
        kap[key] = float(cohen_kappa_score(argmaxes[a], argmaxes[b]))
        mi[key] = float(mutual_info_score(argmaxes[a], argmaxes[b]))
    both_wrong = {}
    for a, b in combinations(MODELS, 2):
        wa = argmaxes[a] != y_te
        wb = argmaxes[b] != y_te
        both_wrong[f"{a}|{b}"] = float((wa & wb).mean())
    A["diversity"] = {"prob_correlation": corr, "disagreement": disag,
                      "kappa": kap, "mutual_info": mi,
                      "both_wrong_rate": both_wrong}

    # ---------- 转诊分析（θ 在 Meta-Cal 上选，Test 上评） ----------
    p_cal_mc = apply_temperature(p_final_mc, T)
    ref = {
        "thresholds": {},
        "metacal_probability_source": (
            "temperature-scaled patient-grouped OOF probabilities from "
            "the selected final candidate"
        ),
    }
    for th in C.REFERRAL_THRESHOLDS:
        conf_mc = p_cal_mc.max(1)
        auto_mc = conf_mc >= th
        conf_te = p_cal_te.max(1)
        auto = conf_te >= th
        yp = p_cal_te.argmax(1)
        row = {
            "metacal_coverage": float(auto_mc.mean()),
            "metacal_auto_acc": float(
                p_cal_mc.argmax(1)[auto_mc].__eq__(
                    y_mc[auto_mc]).mean()) if auto_mc.any() else None,
            "coverage": float(auto.mean()),
            "auto_accuracy": float((yp[auto] == y_te[auto]).mean())
            if auto.any() else None,
            "referred_accuracy": float((yp[~auto] == y_te[~auto]).mean())
            if (~auto).any() else None,
            "referred_class_dist": {
                C.CLASSES[c]: int(((y_te == c) & ~auto).sum())
                for c in range(C.NUM_CLASSES)},
            "per_class_referral_rate": {
                C.CLASSES[c]: float((~auto)[y_te == c].mean())
                for c in range(C.NUM_CLASSES)},
        }
        ref["thresholds"][str(th)] = row
    # θ 选择规则：Meta-Cal 上 auto 准确率 ≥0.90 的最低阈值
    chosen = None
    for th in C.REFERRAL_THRESHOLDS:
        r = ref["thresholds"][str(th)]
        if r["metacal_auto_acc"] and r["metacal_auto_acc"] >= 0.90:
            chosen = th
            break
    ref["chosen_threshold"] = chosen
    ref["selection_rule"] = ("lowest threshold with metacal autonomous "
                             "accuracy >= 0.90")
    A["referral"] = ref

    # ---------- McNemar（最终系统 vs 最佳单骨干 TTA1） ----------
    best_single, single_metacal_scores = select_best_single_from_metacal(
        y_mc, mc_tta1)
    A["best_single_selection"] = {
        "selected": best_single,
        "criterion": "highest meta-calibration macro-F1 at TTA×1",
        "metacal_macro_f1": single_metacal_scores,
        "test_labels_used_for_selection": False,
    }
    A["mcnemar_final_vs_best_single"] = {
        "best_single": best_single,
        "selection_source": "meta-calibration macro-F1 at TTA×1",
        **mcnemar(y_te, p_final_te, tta_mean(te[best_single][0], 1)),
    }

    # ---------- 最终系统 vs 最佳单骨干：配对 Δ 的患者级 bootstrap CI ----------
    # R5.7 直接问"最差类 recall 0.652→0.711 这一提升是否可靠"；McNemar 只
    # 检验全类合并的图像级正确率，答不了"某个具体增益"。此处给 Δ 的区间。
    p_best_te = tta_mean(te[best_single][0], 1)
    delta_fns = {
        "macro_f1": lambda yy, pp: f1_score(yy, pp.argmax(1),
                                            average="macro"),
        "accuracy": lambda yy, pp: accuracy_score(yy, pp.argmax(1)),
        "balanced_accuracy": lambda yy, pp: balanced_accuracy_score(
            yy, pp.argmax(1)),
        "worst_class_recall": lambda yy, pp: float(min(
            recall_score(yy, pp.argmax(1), average=None, zero_division=0))),
        "macro_ap": lambda yy, pp: macro_ap(yy, pp),
    }
    for c, cls in enumerate(C.CLASSES):
        delta_fns[f"recall_{cls}"] = (
            lambda yy, pp, c=c: recall_score(
                yy, pp.argmax(1), labels=[c], average=None,
                zero_division=0)[0])
    A["delta_vs_best_single_ci"] = {
        "best_single": best_single,
        "note": ("Δ = final system − best single backbone (TTA×1), paired "
                 "patient-level bootstrap; worst_class_recall compares the "
                 "minimum over classes within each resample, so the "
                 "identity of the worst class may differ between systems "
                 "— per-class recall deltas are reported alongside."),
        "deltas": paired_bootstrap_delta(y_te, p_final_te, p_best_te,
                                         pts_te, delta_fns, n_boot=500),
    }

    # ---------- 混淆矩阵（行归一） ----------
    cm = confusion_matrix(y_te, p_final_te.argmax(1),
                          labels=list(range(C.NUM_CLASSES)))
    A["confusion_matrix"] = {
        "labels": C.CLASSES, "counts": cm.tolist(),
        "row_normalized": (cm / cm.sum(1, keepdims=True)).round(4).tolist(),
    }

    # ---------- 逐类 PR 曲线点位（Supp Fig 2 数据源） ----------
    from sklearn.metrics import precision_recall_curve
    prc = {}
    for c, cls in enumerate(C.CLASSES):
        pre, rec, _ = precision_recall_curve(
            (y_te == c).astype(int), p_final_te[:, c])
        step = max(1, len(pre) // 200)
        prc[cls] = {"precision": pre[::step].round(4).tolist(),
                    "recall": rec[::step].round(4).tolist()}
    A["pr_curves"] = prc

    # ---------- 活检级聚合（按活检文件夹；标签=该活检的诊断, R4.4/R4.8） ----------
    def aggregate(group_keys):
        rows_by = {}
        for i, k in enumerate(group_keys):
            rows_by.setdefault(k, []).append(i)
        ys, pm, pv, mixed = [], [], [], 0
        for _k, idxs in rows_by.items():
            labs = set(int(v) for v in y_te[idxs])
            if len(labs) > 1:
                mixed += 1        # 跨类分组（重复活检不同诊断）不聚合评估
                continue
            ys.append(labs.pop())
            pm.append(p_cal_te[idxs].mean(0))
            votes = np.bincount(p_cal_te[idxs].argmax(1),
                                minlength=C.NUM_CLASSES)
            pv.append(votes / votes.sum())
        return (np.array(ys), np.vstack(pm), np.vstack(pv),
                len(rows_by), mixed)

    y_b, pm_b, pv_b, n_b, _ = aggregate(pts_te)
    persons_te = [str(x).split("_")[0] for x in pts_te]
    y_p, pm_p, pv_p, n_p, n_mixed = aggregate(persons_te)
    patient_mean_metrics = full_metrics(y_p, pm_p)
    patient_vote_metrics = full_metrics(y_p, pv_p)
    n_patients_evaluated = sum(
        row["n"] for row in patient_mean_metrics["per_class"].values())
    if n_patients_evaluated != len(y_p):
        raise RuntimeError("患者级 per-class 样本数与聚合数组不一致")
    if n_patients_evaluated + n_mixed != n_p:
        raise RuntimeError("患者总数不等于已评估与混合诊断排除数之和")
    A["patient_level"] = {
        "n_biopsies_test": int(n_b),
        "n_patients_test": int(n_p),
        "n_patients_evaluated": int(n_patients_evaluated),
        "n_patients_mixed_diagnoses_excluded": int(n_mixed),
        "biopsy_mean_prob_aggregation": full_metrics(y_b, pm_b),
        "biopsy_majority_vote_aggregation": full_metrics(y_b, pv_b),
        "mean_prob_aggregation": patient_mean_metrics,
        "majority_vote_aggregation": patient_vote_metrics,
    }

    # ---------- 决策曲线分析（一对多净获益, 校准后概率） ----------
    dca = {}
    p_ts = np.linspace(0.05, 0.60, 23)
    for c, cls in enumerate(C.CLASSES):
        yb = (y_te == c).astype(int)
        prev = float(yb.mean())
        pc = p_cal_te[:, c]
        rows = []
        for pt in p_ts:
            pred_pos = pc >= pt
            tp = float((pred_pos & (yb == 1)).mean())
            fp = float((pred_pos & (yb == 0)).mean())
            nb = tp - fp * pt / (1 - pt)
            nb_all = prev - (1 - prev) * pt / (1 - pt)
            rows.append({"pt": round(float(pt), 3), "nb_model": nb,
                         "nb_treat_all": nb_all, "nb_treat_none": 0.0})
        dca[cls] = {"prevalence": prev, "curve": rows}
    A["dca"] = dca

    # ---------- 推理时延汇总 ----------
    ms1 = float(np.mean([timing[m] for m in MODELS]))
    A["inference_cost"] = {
        "per_backbone_ms_per_image": {m: round(timing[m], 3)
                                      for m in MODELS},
        "ensemble_tta1_ms_per_image": round(ms1 * len(MODELS), 2),
        "ensemble_tta8_ms_per_image": round(ms1 * len(MODELS) * 8, 2),
        "note": "单张 GB10、batch=64 吞吐折算；含 softmax 不含 JPEG 解码",
    }

    (C.OUT_DIR / "analysis.json").write_text(
        json.dumps(A, indent=1), encoding="utf-8")
    print("final =", final_key)
    print(json.dumps(A["ensembles"][final_key], indent=1)[:1200])
    print("T =", T, "| ECE", A["calibration"]["before"]["ece"], "->",
          A["calibration"]["after"]["ece"])


if __name__ == "__main__":
    main()
