"""单骨干微调（真实训练，全程记录可复现细节）.

用法:
  python train_backbone.py --model swin_base                     # 主配置
  python train_backbone.py --model swin_base --ablate ce_plain   # 消融臂
  python train_backbone.py --model resnet50 --baseline           # 基线

输出:
  big/checkpoints/<tag>.pt
  results/train_log_<tag>.json   (逐 epoch 指标/耗时/GPU利用率/真实超参)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

import config as C
from dataset import RamDataset, class_counts, load_subset


# ---------------- GPU 利用率采样线程 ----------------
class GpuMonitor(threading.Thread):
    def __init__(self, interval: float = 5.0):
        super().__init__(daemon=True)
        self.interval, self.samples, self._stop = interval, [], False

    def run(self):
        while not self._stop:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip()
                self.samples.append(float(out.split("\n")[0]))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.interval)

    def stop(self) -> dict:
        self._stop = True
        s = self.samples or [0.0]
        return {"mean": float(np.mean(s)), "median": float(np.median(s)),
                "p10": float(np.percentile(s, 10)),
                "p90": float(np.percentile(s, 90)), "n": len(s)}


# ---------------- 损失 ----------------
def soft_target_loss(logits: torch.Tensor, target: torch.Tensor,
                     gamma: float, smoothing: float,
                     alpha: torch.Tensor | None) -> torch.Tensor:
    """支持软标签(MixUp)的 Focal + label smoothing；gamma=0 退化为 CE."""
    n_cls = logits.size(1)
    if target.dim() == 1:
        t = F.one_hot(target, n_cls).float()
    else:
        t = target
    if smoothing > 0:
        t = t * (1 - smoothing) + smoothing / n_cls
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    w = (1 - p).pow(gamma) if gamma > 0 else torch.ones_like(p)
    if alpha is not None:
        w = w * alpha.to(logits.device)[None, :]
    return -(t * w * logp).sum(1).mean()


def mixup_cutmix(x: torch.Tensor, y: torch.Tensor, n_cls: int,
                 rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """交替 MixUp/CutMix，各 p=mix_prob/2."""
    hp = C.HPARAMS
    t = F.one_hot(y, n_cls).float()
    r = rng.random()
    if r < hp["mix_prob"] / 2:                                   # MixUp
        lam = float(rng.beta(hp["mixup_alpha"], hp["mixup_alpha"]))
        idx = torch.randperm(x.size(0), device=x.device)
        x = lam * x + (1 - lam) * x[idx]
        t = lam * t + (1 - lam) * t[idx]
    elif r < hp["mix_prob"]:                                     # CutMix
        lam = float(rng.beta(hp["cutmix_alpha"], hp["cutmix_alpha"]))
        idx = torch.randperm(x.size(0), device=x.device)
        h, w = x.shape[2:]
        rh, rw = int(h * (1 - lam) ** 0.5), int(w * (1 - lam) ** 0.5)
        cy, cx = int(rng.integers(h)), int(rng.integers(w))
        y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, h)
        x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, w)
        x[:, :, y1:y2, x1:x2] = x[idx][:, :, y1:y2, x1:x2]
        lam_adj = 1 - (y2 - y1) * (x2 - x1) / (h * w)
        t = lam_adj * t + (1 - lam_adj) * t[idx]
    return x, t


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    preds, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        preds.append(out.float().argmax(1).cpu().numpy())
        ys.append(np.asarray(y))
    yp, yt = np.concatenate(preds), np.concatenate(ys)
    return f1_score(yt, yp, average="macro"), yp, yt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ablate", default=None, choices=list(C.ABLATIONS))
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    hp = dict(C.HPARAMS)
    abl = C.ABLATIONS[args.ablate] if args.ablate else {
        "sampler": True, "focal": True, "mix": True, "cb_reweight": False}
    tag = (args.model + (f"_abl_{args.ablate}" if args.ablate else "")
           + (f"_seed{args.seed}" if args.seed != C.SEED else ""))
    max_epochs = args.epochs or (
        C.ABLATION_EPOCH_BUDGET if args.ablate else hp["max_epochs"])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    model_name = (C.BASELINE_BACKBONE.get(args.model)
                  or C.BACKBONES[args.model])

    import timm

    model = timm.create_model(model_name, pretrained=True,
                              num_classes=C.NUM_CLASSES).to(device)
    if "efficientnet" in args.model:
        hp["batch_size"] = hp["batch_size_effnet"]

    print(f"[{tag}] loading data into RAM ...", flush=True)
    tr_imgs, tr_lab, _ = load_subset("train")
    mc_imgs, mc_lab, _ = load_subset("metacal")
    counts = class_counts(tr_lab, C.NUM_CLASSES)

    train_ds = RamDataset(tr_imgs, tr_lab, train=True, seed=args.seed)
    mc_ds = RamDataset(mc_imgs, mc_lab, train=False)
    if abl["sampler"]:
        w = (1.0 / counts)[tr_lab]
        sampler = WeightedRandomSampler(
            torch.as_tensor(w, dtype=torch.double), len(tr_lab))
        train_ld = DataLoader(train_ds, hp["batch_size"], sampler=sampler,
                              num_workers=0, drop_last=True)
    else:
        train_ld = DataLoader(train_ds, hp["batch_size"], shuffle=True,
                              num_workers=0, drop_last=True)
    mc_ld = DataLoader(mc_ds, 64, shuffle=False, num_workers=0)

    # Cui et al. 类平衡权重（仅消融臂使用）
    alpha = None
    if abl["cb_reweight"]:
        beta = 0.9999
        eff = (1 - np.power(beta, counts)) / (1 - beta)
        a = (1 / eff) * C.NUM_CLASSES / (1 / eff).sum()
        alpha = torch.tensor(a, dtype=torch.float32)

    head_kw = {"fc", "head", "classifier"}
    head, body = [], []
    for n, p in model.named_parameters():
        (head if any(k in n.lower() for k in head_kw) else body).append(p)
    opt = torch.optim.AdamW(
        [{"params": body, "lr": hp["base_lr"] * hp["backbone_lr_mult"]},
         {"params": head, "lr": hp["base_lr"]}],
        weight_decay=hp["weight_decay"], betas=(0.9, 0.999))
    steps_per_epoch = len(train_ld)
    total_steps = max_epochs * steps_per_epoch
    warm = int(total_steps * hp["warmup_frac"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(warm, 1) if s < warm
        else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(total_steps - warm, 1))))

    gamma = hp["focal_gamma"] if abl["focal"] else 0.0
    rng = np.random.default_rng(args.seed)
    # 1-Hz 采样：与响应信 R4.8"1-Hz nvidia-smi sampling throughout
    # training"及 Methods 声明口径一致（默认 5 s 会构成不实陈述）
    mon = GpuMonitor(interval=1.0)
    mon.start()
    best_f1, best_ep, wait = -1.0, -1, 0
    log_rows = []
    t_start = time.time()
    # ---- epoch 级断点续训 ----
    # 每个 epoch 末落盘完整训练态（权重/优化器/调度器/早停计数/日志/耗时/
    # GPU 采样）；崩溃或被杀后自动从下一 epoch 续跑，绝不从头重训。
    resume_path = C.CKPT_DIR / f"{tag}__resume.pt"
    prior_seconds, prior_gpu = 0.0, []
    start_ep = 1
    if resume_path.exists():
        try:
            rs = torch.load(resume_path, map_location="cpu",
                            weights_only=False)
            model.load_state_dict(rs["model"])
            model.to(device)
            opt.load_state_dict(rs["opt"])
            sched.load_state_dict(rs["sched"])
            best_f1 = rs["best_f1"]
            best_ep = rs["best_ep"]
            wait = rs["wait"]
            log_rows = rs["log_rows"]
            prior_seconds = rs.get("elapsed_seconds", 0.0)
            prior_gpu = rs.get("gpu_samples", [])
            # 随机数状态一并恢复，使续训后的增广/采样序列接续而非重置
            # （旧格式续训点无这些键，.get 兜底，向后兼容）
            if rs.get("np_rng_state") is not None:
                rng.bit_generator.state = rs["np_rng_state"]
            if rs.get("torch_rng_state") is not None:
                torch.set_rng_state(
                    rs["torch_rng_state"].to(torch.uint8).cpu())
            if (rs.get("cuda_rng_state") is not None
                    and torch.cuda.is_available()):
                torch.cuda.set_rng_state(
                    rs["cuda_rng_state"].to(torch.uint8).cpu())
            # start_ep = 已完成 epoch + 1。续训点只在 epoch 完整结束
            # （训完全部 batch + 评估 + 落最优权重）之后原子写入，
            # 故中途崩溃的那个 epoch 永远不会被误判为已完成。
            start_ep = rs["epoch"] + 1
            print(f"[{tag}] RESUME at ep{start_ep} "
                  f"(done {rs['epoch']}, best mcF1={best_f1:.4f} "
                  f"@ep{best_ep}, prior {prior_seconds/3600:.2f}h)",
                  flush=True)
        except Exception as e:  # noqa: BLE001  损坏的续训点不阻塞训练
            print(f"[{tag}] RESUME FAILED ({e!r}) — 从头训练", flush=True)
            start_ep = 1
    if start_ep > max_epochs:
        print(f"[{tag}] 续训点显示已达 max_epochs，直接收尾", flush=True)
    for ep in range(start_ep, max_epochs + 1):
        model.train()
        t0, run_loss, nb = time.time(), 0.0, 0
        for x, y in train_ld:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if abl["mix"]:
                x, t = mixup_cutmix(x, y, C.NUM_CLASSES, rng)
            else:
                t = y
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = soft_target_loss(out.float(), t, gamma,
                                        hp["label_smoothing"], alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item()
            nb += 1
        f1, _, _ = evaluate(model, mc_ld, device)
        dt = time.time() - t0
        log_rows.append({"epoch": ep, "loss": run_loss / max(nb, 1),
                         "metacal_macro_f1": f1, "epoch_seconds": dt})
        print(f"[{tag}] ep{ep:02d} loss={run_loss/max(nb,1):.4f} "
              f"mcF1={f1:.4f} ({dt:.0f}s)", flush=True)
        if f1 > best_f1:
            best_f1, best_ep, wait = f1, ep, 0
            # 最优权重同样原子写：predict.py 依赖它，写一半即不可加载
            best_path = C.CKPT_DIR / f"{tag}.pt"
            btmp = best_path.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "metacal_macro_f1": f1, "model_name": model_name},
                       btmp)
            btmp.replace(best_path)
        else:
            wait += 1
        # ---- 断点落盘：仅在本 epoch 全部 batch 训完 + 评估完 + 最优权重
        # 落盘之后执行，故 "epoch": ep 严格表示"第 ep 轮已完成"；
        # 中途崩溃时本次不写，恢复后自然重跑该 epoch。原子写防写一半掉电。
        tmp = resume_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep,
                    "best_f1": best_f1, "best_ep": best_ep, "wait": wait,
                    "log_rows": log_rows,
                    "elapsed_seconds": prior_seconds + (time.time()
                                                        - t_start),
                    "gpu_samples": prior_gpu + list(mon.samples),
                    "np_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (torch.cuda.get_rng_state()
                                       if torch.cuda.is_available()
                                       else None)}, tmp)
        tmp.replace(resume_path)
        if wait >= hp["early_stop_patience"]:
            print(f"[{tag}] early stop at ep{ep} (best ep{best_ep})",
                  flush=True)
            break

    gpu_stats = mon.stop()
    log = {
        "tag": tag, "model_name": model_name, "ablate": args.ablate,
        "hparams": hp, "ablation_flags": abl, "seed": args.seed,
        "max_epochs": max_epochs, "stopped_epoch": log_rows[-1]["epoch"],
        "best_epoch": best_ep, "best_metacal_macro_f1": best_f1,
        # 含续训前累计时长/采样，跨中断仍是真实总量（R4.8 口径）
        "total_seconds": round(prior_seconds + time.time() - t_start, 1),
        "resumed": start_ep > 1,
        "mean_epoch_seconds": round(float(np.mean(
            [r["epoch_seconds"] for r in log_rows])), 1),
        "gpu_utilization": gpu_stats,
        # 原始采样序列一并写入——正文/响应信承诺 "utilisation samples
        # are released"，只留汇总会构成空头承诺
        "gpu_utilization_samples": [int(round(v))
                                    for v in (prior_gpu + list(mon.samples))],
        "train_images": int(len(tr_lab)), "metacal_images": int(len(mc_lab)),
        "class_counts_train": counts.tolist(),
        "epochs": log_rows, "env": C.env_report(),
    }
    (C.OUT_DIR / f"train_log_{tag}.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8")
    resume_path.unlink(missing_ok=True)      # 正常收尾即清理续训点
    print(f"[{tag}] DONE best mcF1={best_f1:.4f} @ep{best_ep} "
          f"total={log['total_seconds']/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
