"""传统 ML 基线（回应 R3.2）：ImageNet 冻结特征 + LR / 随机森林.

Swin-B (ImageNet-22k) 冻结特征 (GAP 1024 维) → LogisticRegression / RF。
输出: results/frozen_baseline.json
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import config as C
from dataset import RamDataset, load_subset


@torch.no_grad()
def extract(model, imgs, labels, batch=64) -> np.ndarray:
    ds = RamDataset(imgs, labels, train=False)
    ld = torch.utils.data.DataLoader(ds, batch, shuffle=False)
    feats = []
    for x, _ in ld:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model(x.to("cuda"))
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats)


def main() -> None:
    import timm

    model = timm.create_model(C.BACKBONES["swin_base"], pretrained=True,
                              num_classes=0)   # 特征模式
    model.eval().to("cuda")

    out = {}
    t0 = time.time()
    tr_imgs, tr_lab, _ = load_subset("train")
    te_imgs, te_lab, _ = load_subset("test")
    Xtr = extract(model, tr_imgs, tr_lab)
    Xte = extract(model, te_imgs, te_lab)
    del tr_imgs, te_imgs
    out["feature_dim"] = int(Xtr.shape[1])
    out["extract_seconds"] = round(time.time() - t0, 1)

    for name, clf in {
        "logreg": LogisticRegression(C=1.0, max_iter=3000, n_jobs=-1),
        "random_forest": RandomForestClassifier(
            n_estimators=500, n_jobs=-1, random_state=C.SEED,
            class_weight="balanced"),
    }.items():
        t1 = time.time()
        clf.fit(Xtr, tr_lab)
        p = clf.predict_proba(Xte)
        out[name] = {
            "accuracy": accuracy_score(te_lab, p.argmax(1)),
            "macro_f1": f1_score(te_lab, p.argmax(1), average="macro"),
            "auc_ovr_macro": roc_auc_score(te_lab, p, multi_class="ovr",
                                           average="macro"),
            "fit_seconds": round(time.time() - t1, 1),
        }
        print(name, out[name], flush=True)

    (C.OUT_DIR / "frozen_baseline.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
