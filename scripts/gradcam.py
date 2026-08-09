"""Grad-CAM 可解释性分析（回应 R2.3 / R3.5 / R5.15）.

对 ConvNeXt-B（卷积语义最清晰）做 Grad-CAM：
  · 每类 2 例高置信正确样本
  · 少数类被误判为 IgAN 的代表样本（每个受累类最多 3 例）
输出: results/gradcam/<case_id>.png + results/gradcam_index.json
"""
from __future__ import annotations

import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from dataset import RamDataset, load_subset

GC_DIR = C.OUT_DIR / "gradcam"
GC_DIR.mkdir(exist_ok=True)
MODEL_KEY = "convnext_base"


class GradCam:
    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module):
        self.acts, self.grads = None, None
        layer.register_forward_hook(self._fh)
        layer.register_full_backward_hook(self._bh)
        self.model = model

    def _fh(self, _m, _i, o):
        self.acts = o.detach()

    def _bh(self, _m, _gi, go):
        self.grads = go[0].detach()

    def __call__(self, x: torch.Tensor, cls: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        out[0, cls].backward()
        a, g = self.acts[0], self.grads[0]          # C,H,W
        wgt = g.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((wgt * a).sum(0))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy()


def main() -> None:
    import timm

    ck = torch.load(C.CKPT_DIR / f"{MODEL_KEY}.pt", map_location="cpu",
                    weights_only=False)
    model = timm.create_model(ck["model_name"], pretrained=False,
                              num_classes=C.NUM_CLASSES)
    model.load_state_dict(ck["model"])
    model.eval().to("cuda")
    layer = model.stages[-1]          # ConvNeXt 最后 stage 输出
    gc = GradCam(model, layer)

    z = np.load(C.PRED_DIR / f"{MODEL_KEY}__test.npz")
    probs = z["probs"][0].astype(np.float32)      # TTA0
    labels, pts = z["labels"], z["patients"]
    yp, conf = probs.argmax(1), probs.max(1)

    imgs, lab2, rows = load_subset("test")
    ds = RamDataset(imgs, lab2, train=False)

    picks = []          # (idx, kind)
    igan = C.CLASS_TO_IDX["IgAN"]
    for c in range(C.NUM_CLASSES):
        ok = np.where((labels == c) & (yp == c))[0]
        ok = ok[np.argsort(-conf[ok])][:2]
        picks += [(int(i), "correct") for i in ok]
        if c != igan:
            mis = np.where((labels == c) & (yp == igan))[0]
            mis = mis[np.argsort(-conf[mis])][:3]
            picks += [(int(i), "missed_as_IgAN") for i in mis]

    index = []
    for idx, kind in picks:
        x, _y = ds[idx]
        x = x[None].to("cuda")
        cam_pred = gc(x, int(yp[idx]))
        raw = imgs[idx]
        cam_img = np.array(
            torch.nn.functional.interpolate(
                torch.tensor(cam_pred)[None, None], size=(384, 384),
                mode="bilinear", align_corners=False)[0, 0])
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 4), dpi=160)
        axes[0].imshow(raw)
        axes[0].set_title(
            f"true={C.CLASSES[labels[idx]]}  "
            f"pred={C.CLASSES[yp[idx]]}  p={conf[idx]:.2f}", fontsize=9)
        axes[1].imshow(raw)
        axes[1].imshow(cam_img, cmap="jet", alpha=0.4)
        axes[1].set_title("Grad-CAM (predicted class)", fontsize=9)
        for ax in axes:
            ax.axis("off")
        cid = f"{kind}_{C.CLASSES[labels[idx]]}_{idx}"
        fig.tight_layout()
        fig.savefig(GC_DIR / f"{cid}.png", bbox_inches="tight")
        plt.close(fig)
        index.append({"case": cid, "kind": kind,
                      "true": C.CLASSES[labels[idx]],
                      "pred": C.CLASSES[yp[idx]],
                      "conf": float(conf[idx]), "patient": str(pts[idx])})
        print("saved", cid, flush=True)

    (C.OUT_DIR / "gradcam_index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
