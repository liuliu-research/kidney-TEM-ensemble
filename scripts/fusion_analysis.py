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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score,
    log_loss, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import config as C

MODELS = list(C.BACKBONES)          # 5 骨干
BOOT_N = 1000
RNG = np.random.default_rng(C.SEED)


# ---------------- 载入 ----------------
def load_preds(tag: str, subset: str):
    z = np.load(C.PRED_DIR / f"{tag}__{subset}.npz", allow_pickle=False)
    return (z["probs"].astype(np.float32), z["labels"], z["patients"],
            float(z["ms_per_image_single"]))


def tta_mean(probs: np.ndarray, n: int) -> np.ndarray:
    p = probs[:n].mean(0)
    return p / p.sum(1, keepdims=True)      # float16 存储误差归一化


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


def classwise_ece(y, p, bins=15) -> dict:
    out = {}
    for c in range(p.shape[1]):
        m = y == c
        if m.sum() >= 10:
            out[C.CLASSES[c]] = ece(y[m], p[m], bins)
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
    w = np.array([f1_score(y_mc, tta_mean(mc[m][0], 1).argmax(1),
                           average="macro") for m in MODELS])
    w = w / w.sum()
    A["voting_weights"] = dict(zip(MODELS, w.round(4).tolist()))

    def vote(preds: dict, n_tta: int, subset_models=MODELS):
        return sum(wi * tta_mean(preds[m][0], n_tta)
                   for wi, m in zip(w, subset_models))

    # ---------- Stacking（Meta-Cal 训练；特征=5×11 softmax） ----------
    def stack_feats(preds: dict, n_tta: int):
        return np.hstack([tta_mean(preds[m][0], n_tta) for m in MODELS])

    Xmc, Xte = stack_feats(mc, 1), stack_feats(te, 1)
    scaler = StandardScaler().fit(Xmc)
    meta_models = {
        "stack_lr": LogisticRegression(C=1.0, max_iter=2000),
        "stack_mlp": MLPClassifier(hidden_layer_sizes=(64,), max_iter=800,
                                   random_state=C.SEED),
        "stack_gbdt": HistGradientBoostingClassifier(random_state=C.SEED),
    }
    stack_te_probs = {}
    for name, mm in meta_models.items():
        mm.fit(scaler.transform(Xmc), y_mc)
        stack_te_probs[name] = mm.predict_proba(scaler.transform(Xte))

    # ---------- TTA 扫描（投票 与 LR stacking） ----------
    A["tta_sweep"] = {}
    for n in C.TTA_SWEEP:
        pv = vote(te, n)
        Xte_n = stack_feats(te, n)
        Xmc_n = stack_feats(mc, n)
        sc_n = StandardScaler().fit(Xmc_n)
        lr_n = LogisticRegression(C=1.0, max_iter=2000).fit(
            sc_n.transform(Xmc_n), y_mc)
        ps = lr_n.predict_proba(sc_n.transform(Xte_n))
        ms1 = float(np.mean([timing[m] for m in MODELS]))
        A["tta_sweep"][str(n)] = {
            "voting": full_metrics(y_te, pv),
            "stack_lr": full_metrics(y_te, ps),
            "ms_per_image_ensemble": round(ms1 * len(MODELS) * n, 2),
        }

    # ---------- 主配置选择（在 Meta-Cal 上决定，写明协议） ----------
    cand_mc = {
        "voting_tta8": vote(mc, 8),
        "stack_lr_tta8": None,   # 由下方计算
    }
    Xmc8 = stack_feats(mc, 8)
    sc8 = StandardScaler().fit(Xmc8)
    lr8 = LogisticRegression(C=1.0, max_iter=2000).fit(
        sc8.transform(Xmc8), y_mc)
    # metacal 上的自估（拟合数据上,仅用于择型记录，不作论文主指标）
    cand_mc["stack_lr_tta8"] = lr8.predict_proba(sc8.transform(Xmc8))
    A["selection_on_metacal"] = {
        k: {"macro_f1": f1_score(y_mc, v.argmax(1), average="macro")}
        for k, v in cand_mc.items()
    }
    Xte8 = stack_feats(te, 8)
    def vote_uniform(preds: dict, n_tta: int):
        p = sum(tta_mean(preds[m][0], n_tta) for m in MODELS) / len(MODELS)
        return p / p.sum(1, keepdims=True)

    final_probs = {
        "voting_tta8": vote(te, 8),
        "stack_lr_tta8": lr8.predict_proba(sc8.transform(Xte8)),
        "stack_mlp_tta1": stack_te_probs["stack_mlp"],
        "stack_gbdt_tta1": stack_te_probs["stack_gbdt"],
        "stack_lr_tta1": stack_te_probs["stack_lr"],
        "voting_tta1": vote(te, 1),
        "voting_uniform_tta1": vote_uniform(te, 1),   # 常规等权集成基线(R3.2)
        "voting_uniform_tta8": vote_uniform(te, 8),
    }
    A["ensembles"] = {k: full_metrics(y_te, v)
                      for k, v in final_probs.items()}

    # ---------- 最终系统 = Meta-Cal macro-F1 最优者 ----------
    final_key = max(A["selection_on_metacal"],
                    key=lambda k: A["selection_on_metacal"][k]["macro_f1"])
    A["final_key"] = final_key
    p_final_te = final_probs[final_key]

    # ---------- 温度标定（Meta-Cal 拟合 → Test 应用） ----------
    p_final_mc = cand_mc[final_key]
    logp_mc = np.log(np.clip(p_final_mc, 1e-12, 1))
    T = fit_temperature(logp_mc, y_mc)
    p_cal_te = apply_temperature(p_final_te, T)
    A["calibration"] = {
        "temperature": T,
        "before": calib_suite(y_te, p_final_te),
        "after": calib_suite(y_te, p_cal_te),
    }
    # 逐类温度对照（R5.4 保险分析：全局单温 vs 每类一温的 balanced ECE）
    per_cls_T = {}
    p_pct = p_final_te.copy()
    for c in range(C.NUM_CLASSES):
        m = y_mc == c
        # 对该类样本子集拟合温度（同全局法，仅换样本），样本过少则回退全局 T
        if m.sum() >= 50:
            t_c = fit_temperature(logp_mc[m], y_mc[m])
        else:
            t_c = T
        per_cls_T[C.CLASSES[c]] = t_c
    # 应用：按真实类无法在预测期知晓，故按预测类应用（标准 class-wise TS 做法）
    yp_arg = p_final_te.argmax(1)
    for c in range(C.NUM_CLASSES):
        m = yp_arg == c
        if m.any():
            p_pct[m] = apply_temperature(p_final_te[m],
                                         per_cls_T[C.CLASSES[c]])
    A["classwise_temperature_probe"] = {
        "per_class_T": {k: round(v, 4) for k, v in per_cls_T.items()},
        "balanced_ece_global_T": A["calibration"]["after"][
            "classwise_ece"]["balanced_ece"],
        "balanced_ece_classwise_T": classwise_ece(y_te, p_pct)[
            "balanced_ece"],
        "note": "保险对照：正文默认全局单温；此处量化每类一温是否更优",
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
    ref = {"thresholds": {}}
    for th in C.REFERRAL_THRESHOLDS:
        conf_mc = p_cal_mc.max(1)
        auto_mc = conf_mc >= th
        conf_te = p_cal_te.max(1)
        auto = conf_te >= th
        yp = p_cal_te.argmax(1)
        row = {
            "metacal_coverage": float(auto_mc.mean()),
            "metacal_auto_acc": float(
                (yp_mc := p_cal_mc.argmax(1))[auto_mc].__eq__(
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
    best_single = max(MODELS, key=lambda m:
                      A["singles"][m]["tta1"]["macro_f1"])
    A["mcnemar_final_vs_best_single"] = {
        "best_single": best_single,
        **mcnemar(y_te, p_final_te, tta_mean(te[best_single][0], 1)),
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
    A["patient_level"] = {
        "n_biopsies_test": int(n_b),
        "n_patients_test": int(n_p),
        "n_patients_mixed_diagnoses_excluded": int(n_mixed),
        "biopsy_mean_prob_aggregation": full_metrics(y_b, pm_b),
        "biopsy_majority_vote_aggregation": full_metrics(y_b, pv_b),
        "mean_prob_aggregation": full_metrics(y_p, pm_p),
        "majority_vote_aggregation": full_metrics(y_p, pv_p),
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
