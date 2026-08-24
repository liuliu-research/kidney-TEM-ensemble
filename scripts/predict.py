"""对 Meta-Cal / Test 推理，输出 16 变换 TTA softmax 数组 + 逐图耗时.

TTA 变换（确定性顺序，前 N 个即 N-fold）：
  0..7  = D4 群（id, hflip, rot180, hflip∘rot180, rot90, hflip∘rot90,
          rot270, hflip∘rot270）
  8..15 = 同序 × 对比度 0.95
前缀语义: 1=[id], 2=[id,hflip], 4=前4, 8=D4 全, 16=全部。

用法: python predict.py --model swin_base [--ablate KEY] --subset test
输出: big/preds/<tag>__<subset>.npz  (probs: (16,N,C) float16, labels, timing)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import config as C
from dataset import RamDataset, load_subset


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def tta_batch(x: torch.Tensor, k: int) -> torch.Tensor:
    """对 NCHW batch 应用第 k 个确定性 TTA 变换."""
    base = k % 8
    if base in (2, 3):
        x = torch.rot90(x, 2, (2, 3))
    elif base in (4, 5):
        x = torch.rot90(x, 1, (2, 3))
    elif base in (6, 7):
        x = torch.rot90(x, 3, (2, 3))
    if base % 2 == 1:
        x = torch.flip(x, (3,))
    if k >= 8:                       # 像素域对比度 0.95（围绕 0.5 收缩）
        m, s = _MEAN.to(x.device), _STD.to(x.device)
        p = (x * s + m - 0.5) * 0.95 + 0.5
        x = (p.clamp(0, 1) - m) / s
    return x


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ablate", default=None)
    ap.add_argument("--subset", required=True,
                    choices=["metacal", "test"])
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    tag = args.model + (f"_abl_{args.ablate}" if args.ablate else "")
    out_path = C.PRED_DIR / f"{tag}__{args.subset}.npz"
    if out_path.exists():
        # "文件存在即跳过"必须配合完整性校验：若上一趟被杀在 savez 中途，
        # 截断的 npz 会被当成成功结果跳过，下游 ablation_analysis /
        # fusion_analysis 读到 BadZipFile 才炸，且此时已难追溯是哪一趟坏的。
        try:
            with np.load(out_path, allow_pickle=False) as z:
                _ = z["probs"].shape, z["labels"].shape, z["patients"].shape
            print(f"[{tag}/{args.subset}] exists, skip", flush=True)
            return
        except Exception as e:  # noqa: BLE001  截断/损坏 → 重算
            print(f"[{tag}/{args.subset}] 既有文件损坏({e!r})，重算", flush=True)
            out_path.unlink(missing_ok=True)

    import timm

    ck = torch.load(C.CKPT_DIR / f"{tag}.pt", map_location="cpu",
                    weights_only=False)
    model = timm.create_model(ck["model_name"], pretrained=False,
                              num_classes=C.NUM_CLASSES)
    model.load_state_dict(ck["model"])
    model.eval().to("cuda")

    imgs, labels, rows = load_subset(args.subset)
    ds = RamDataset(imgs, labels, train=False)
    ld = torch.utils.data.DataLoader(ds, args.batch, shuffle=False,
                                     num_workers=0)
    n = len(labels)
    probs = np.empty((16, n, C.NUM_CLASSES), dtype=np.float16)
    t_single = None
    for k in range(16):
        t0, i = time.time(), 0
        for x, _ in ld:
            x = tta_batch(x.to("cuda", non_blocking=True), k)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            p = torch.softmax(out.float(), 1).cpu().numpy()
            probs[k, i:i + len(p)] = p.astype(np.float16)
            i += len(p)
        dt = time.time() - t0
        if k == 0:
            t_single = dt
        print(f"[{tag}/{args.subset}] tta{k} {dt:.0f}s", flush=True)

    pts = np.array([r["pt"] for r in rows])
    # 原子落盘，与 train_backbone 的断点写法一致。
    # tmp 名必须以 .npz 结尾——否则 numpy 会自动补后缀，replace() 找不到文件。
    tmp_path = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp_path, probs=probs, labels=labels, patients=pts,
        secs_per_pass=t_single, n_images=n,
        ms_per_image_single=1000 * t_single / n,
    )
    tmp_path.replace(out_path)
    meta = {"tag": tag, "subset": args.subset, "n": int(n),
            "ms_per_image_single_pass": round(1000 * t_single / n, 3)}
    (C.OUT_DIR / f"predict_meta_{tag}__{args.subset}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{tag}/{args.subset}] saved {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
