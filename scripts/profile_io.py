"""数据治理层吞吐剖析（回应 R4.8 / R5.9）：RAM 缓存 vs 直读 TIFF.

各模式训练 EfficientNet-B4 若干步，重复 3 次测：
  · 每步耗时（数据等待 vs 计算分解）
  · GPU 利用率（1 Hz 采样）
  · CPU 利用率（1 Hz 采样，全机口径；R5.9 逐项点名要求）
  · torch.profiler 窗口（30 步）导出算子级摘要
ON  = JPEG-384 RAM 缓存 (num_workers=0)
OFF = 原始 TIFF 磁盘直读 + 在线 resize (num_workers=8)
输出: results/io_profile.json
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config as C
from dataset import RamDataset, load_subset, IMAGENET_MEAN, IMAGENET_STD
from train_backbone import GpuMonitor

STEPS = 150          # 每次重复测量的步数
REPEATS = 3
BATCH = 40

Image.MAX_IMAGE_PIXELS = None


class TiffDataset(Dataset):
    """治理关闭模式：直接读原始 TIFF、在线 resize。"""

    def __init__(self):
        manifest = json.loads((C.OUT_DIR / "manifest.json").read_text("utf-8"))
        split = json.loads((C.OUT_DIR / "split.json").read_text("utf-8"))
        self.rows = [r for r in manifest if split[r["pt"]] == "train"]
        from config import CLASS_TO_IDX
        self.idx = CLASS_TO_IDX

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(r["path"]) as im:
            a = np.asarray(im.convert("RGB").resize((384, 384),
                                                    Image.LANCZOS))
        x = (a.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(x.transpose(2, 0, 1)), self.idx[r["cls"]]


class CpuMonitor:
    """CPU 利用率 1 Hz 采样（全机口径）.

    R5.9 逐项列出的六项里 CPU 利用率原先未测。psutil 缺失时不抛错，
    转而如实记录 available=False——绝不能让"没测到"变成编出来的数字。
    """

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._t: threading.Thread | None = None
        try:
            import psutil
            self._ps = psutil
        except ImportError:
            self._ps = None

    def _run(self) -> None:
        self._ps.cpu_percent(None)        # 首次调用返回无意义值，丢弃
        while not self._stop.is_set():
            self._samples.append(float(self._ps.cpu_percent(self.interval)))

    def start(self) -> None:
        if self._ps is None:
            return
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self) -> dict:
        if self._ps is None:
            return {"available": False,
                    "note": "psutil not installed; CPU utilisation not "
                            "sampled (see RUNBOOK)"}
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=self.interval * 2 + 1)
        s = self._samples
        if not s:
            return {"available": False, "note": "no CPU samples collected"}
        return {
            "available": True,
            "n_samples": len(s),
            "mean": float(np.mean(s)),
            "median": float(np.median(s)),
            "p10": float(np.percentile(s, 10)),
            "p90": float(np.percentile(s, 90)),
            "n_logical_cores": int(self._ps.cpu_count(logical=True)),
        }


def measure(loader, model, opt, steps: int) -> dict:
    it = iter(loader)
    data_t, step_t = [], []
    mon = GpuMonitor(interval=1.0)
    cpu_mon = CpuMonitor(interval=1.0)
    mon.start()
    cpu_mon.start()
    torch.cuda.synchronize()
    for s in range(steps):
        t0 = time.time()
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        t1 = time.time()
        x = x.to("cuda", non_blocking=True)
        y = torch.as_tensor(y).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        data_t.append(t1 - t0)
        step_t.append(time.time() - t0)
    g = mon.stop()
    c = cpu_mon.stop()
    return {
        "steps": steps,
        "mean_step_s": float(np.mean(step_t)),
        "mean_data_wait_s": float(np.mean(data_t)),
        "data_wait_fraction": float(np.sum(data_t) / np.sum(step_t)),
        "imgs_per_s": BATCH / float(np.mean(step_t)),
        "gpu_util": g,
        "cpu_util": c,
    }


def main() -> None:
    import timm

    results = {"batch": BATCH, "steps_per_repeat": STEPS,
               "repeats": REPEATS, "modes": {}}
    for mode in ("governance_on", "governance_off"):
        model = timm.create_model(C.BACKBONES["efficientnet_b4"],
                                  pretrained=True,
                                  num_classes=C.NUM_CLASSES).to("cuda")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        if mode == "governance_on":
            imgs, lab, _ = load_subset("train")
            ds = RamDataset(imgs, lab, train=False)
            ld = DataLoader(ds, BATCH, shuffle=True, num_workers=0,
                            drop_last=True)
        else:
            ds = TiffDataset()
            ld = DataLoader(ds, BATCH, shuffle=True, num_workers=8,
                            prefetch_factor=2, drop_last=True)
        reps = []
        for rep in range(REPEATS):
            r = measure(ld, model, opt, STEPS)
            reps.append(r)
            print(mode, "rep", rep, json.dumps(
                {k: round(v, 3) if isinstance(v, float) else v
                 for k, v in r.items() if k != "gpu_util"}), flush=True)
        n_train = len(ds)
        mean_step = float(np.mean([r["mean_step_s"] for r in reps]))
        cpu_means = [r["cpu_util"]["mean"] for r in reps
                     if r["cpu_util"].get("available")]
        results["modes"][mode] = {
            "repeats": reps,
            "mean_step_s": mean_step,
            "extrapolated_epoch_minutes": round(
                mean_step * (n_train // BATCH) / 60, 1),
            "gpu_util_mean": float(np.mean(
                [r["gpu_util"]["mean"] for r in reps])),
            "cpu_util_mean": (float(np.mean(cpu_means))
                              if cpu_means else None),
        }
        # profiler 窗口（各模式一次，导出摘要文本）
        from torch.profiler import profile, ProfilerActivity
        it = iter(ld)
        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as prof:
            for _ in range(30):
                try:
                    x, y = next(it)
                except StopIteration:
                    it = iter(ld)
                    x, y = next(it)
                x = x.to("cuda")
                y = torch.as_tensor(y).to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = torch.nn.functional.cross_entropy(model(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        (C.OUT_DIR / f"profiler_{mode}.txt").write_text(
            prof.key_averages().table(sort_by="cpu_time_total",
                                      row_limit=25), encoding="utf-8")
        del model, opt, ds, ld
        torch.cuda.empty_cache()

    on = results["modes"]["governance_on"]
    off = results["modes"]["governance_off"]
    results["speedup_step"] = round(off["mean_step_s"] / on["mean_step_s"], 2)
    (C.OUT_DIR / "io_profile.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print("speedup =", results["speedup_step"], flush=True)


if __name__ == "__main__":
    main()
