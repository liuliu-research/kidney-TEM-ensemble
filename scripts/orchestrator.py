"""总调度：CPU 阶段立即执行；GPU 阶段等 R5 训练结束且 GPU 空闲后按序执行.

纪律：
  · 绝不干扰他人进程；启动时记录现存 GPU 训练 PID，只在其全部退出
    且 GPU 利用率连续 3 次(每 10 min)低于阈值后才开始 GPU 步骤。
  · 每步有 marker（results/markers/<step>.done），断点续跑安全。
  · 心跳写 logs/heartbeat.log。

用法: setsid nohup python orchestrator.py > logs/orchestrator.out 2>&1 &
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import config as C

MARK = C.OUT_DIR / "markers"
MARK.mkdir(exist_ok=True)
PY = sys.executable
HERE = Path(__file__).parent


def beat(msg: str) -> None:
    line = f"{datetime.now():%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(C.LOG_DIR / "heartbeat.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(name: str, args: list[str], retries: int = 1) -> None:
    marker = MARK / f"{name}.done"
    if marker.exists():
        beat(f"SKIP {name} (marker)")
        return
    for attempt in range(retries + 1):
        beat(f"START {name} (attempt {attempt})")
        log = open(C.LOG_DIR / f"{name}.log", "a", encoding="utf-8")
        r = subprocess.run([PY] + args, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        if r.returncode == 0:
            marker.write_text(datetime.now().isoformat())
            beat(f"DONE {name}")
            (MARK / f"{name}.FAILED").unlink(missing_ok=True)
            return
        beat(
            f"FAIL {name} rc={r.returncode} "
            f"(重试 {attempt + 1}/{retries}，训练步骤将从断点续跑)"
        )
        time.sleep(60)
    (MARK / f"{name}.FAILED").write_text(datetime.now().isoformat())
    raise SystemExit(f"step {name} failed")


def pick_ablation_backbone() -> str:
    """按 Meta-Cal macro-F1 选消融骨干；结果落盘供稿件 token 引用.

    只读 5 个主训骨干的 train_log（不含 resnet50 基线、不含 seed 复跑），
    依据是 Meta-Cal 指标，Test 集不参与——不构成选择偏倚。
    """
    out = C.OUT_DIR / "ablation_backbone.json"
    if out.exists():
        return json.loads(out.read_text("utf-8"))["backbone"]
    scores = {}
    for m in C.BACKBONES:
        p = C.OUT_DIR / f"train_log_{m}.json"
        if p.exists():
            scores[m] = json.loads(p.read_text("utf-8"))["best_metacal_macro_f1"]
    if not scores:
        beat("WARN 无主训日志，消融骨干回退默认值")
        return C.ABLATION_BACKBONE_FALLBACK
    best = max(scores, key=scores.get)
    out.write_text(
        json.dumps(
            {
                "backbone": best,
                "criterion": "highest meta-calibration macro-F1 among the five "
                "main backbones; the held-out test set played no "
                "role in this choice",
                "metacal_macro_f1": scores,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return best


def gpu_busy_pids() -> set[int]:
    """他人训练进程 PID（按命令行特征识别；常驻推理服务 uvicorn 不算）."""
    pats = (
        "train_image.py",
        "train_text.py",
        "train_kg.py",
        "train_fusion.py",
        "run_all.py",
        "baselines/",
    )
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,args"], capture_output=True, text=True, timeout=15
        ).stdout
        pids = set()
        for line in out.splitlines():
            if "python" in line and any(p in line for p in pats):
                if "revision_experiments" in line:
                    continue  # 自己人
                pid = line.strip().split()[0]
                if pid.isdigit():
                    pids.add(int(pid))
        return pids
    except Exception:  # noqa: BLE001
        return set()


def gpu_util() -> float:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
        return float(out.strip().split("\n")[0])
    except Exception:  # noqa: BLE001
        return 100.0


def wait_for_gpu() -> None:
    """等待外部训练退出且利用率连续 3 次低于阈值。"""
    if (MARK / "gpu_free.done").exists():
        # gpu_free.done 只证明上一次启动时 GPU 空闲，不能跨主机重启永久信任。
        # 若期间有其他作业自启动，复用旧门闩会直接争卡。
        alive = gpu_busy_pids()
        util = gpu_util()
        if not alive and util < C.GPU_IDLE_UTIL_THRESHOLD:
            beat(f"GPU FREE marker revalidated (util={util})")
            return
        beat("STALE GPU FREE marker ignored: " f"alive={sorted(alive)} util={util}")
    C.FOREIGN_PIDS_FILE.write_text(json.dumps(sorted(gpu_busy_pids())))
    beat(f"WAIT-GPU foreign train pids={C.FOREIGN_PIDS_FILE.read_text()}")
    ok_streak = 0
    while True:
        alive = gpu_busy_pids()  # 每轮重扫，防新训练启动误撞
        util = gpu_util()
        if not alive and util < C.GPU_IDLE_UTIL_THRESHOLD:
            ok_streak += 1
            beat(f"GPU idle check {ok_streak}/3 (util={util})")
            if ok_streak >= 3:
                (MARK / "gpu_free.done").write_text(datetime.now().isoformat())
                beat("GPU FREE — starting GPU stages")
                return
        else:
            ok_streak = 0
            beat(f"GPU busy: alive={sorted(alive)} util={util}")
        time.sleep(C.GPU_IDLE_CHECK_MINUTES * 60)


def main() -> None:
    beat("=== orchestrator start ===")
    # ---------- CPU 阶段 ----------
    run_step("manifest", ["build_manifest.py"])
    run_step("split", ["make_split.py"])
    run_step("cache", ["build_cache.py"])
    run_step("image_quality", ["image_quality.py"])

    # ---------- GPU 阶段 ----------
    wait_for_gpu()
    # 训练/推理步骤 retries=3：配合 train_backbone 的 epoch 级断点续训，
    # 单次崩溃可在同一趟内自愈续跑，不回到从头训练。
    for m in C.BACKBONES:  # 主训 5 骨干
        run_step(f"train_{m}", ["train_backbone.py", "--model", m], retries=3)
    run_step(
        "train_resnet50",
        ["train_backbone.py", "--model", "resnet50", "--baseline"],
        retries=3,
    )
    for m in list(C.BACKBONES) + ["resnet50"]:  # 推理（16×TTA 两子集）
        for sub in ("metacal", "test"):
            run_step(
                f"pred_{m}_{sub}",
                ["predict.py", "--model", m, "--subset", sub],
                retries=3,
            )
    # ---- 消融骨干：按 Meta-Cal macro-F1 选实测最强者（依据不涉 Test） ----
    abl_bb = pick_ablation_backbone()
    beat(f"消融骨干选定: {abl_bb}")
    for abl in C.ABLATIONS:
        run_step(
            f"abl_{abl}",
            ["train_backbone.py", "--model", abl_bb, "--ablate", abl],
            retries=3,
        )
        for sub in ("metacal", "test"):
            run_step(
                f"pred_abl_{abl}_{sub}",
                ["predict.py", "--model", abl_bb, "--ablate", abl, "--subset", sub],
                retries=3,
            )
    run_step("frozen_baseline", ["frozen_baseline.py"])
    run_step("fusion_analysis", ["fusion_analysis.py"])
    run_step("ablation_analysis", ["ablation_analysis.py"])
    run_step("gradcam", ["gradcam.py"])
    run_step("profile_io", ["profile_io.py"])
    beat("=== ALL DONE ===")
    (MARK / "ALL.done").write_text(datetime.now().isoformat())

    # ---------- 可选保险批次（种子重复；失败不阻塞，主线已完成） ----------
    for seed in (7, 2026):
        try:
            run_step(
                f"train_swin_seed{seed}",
                ["train_backbone.py", "--model", "swin_base", "--seed", str(seed)],
                retries=0,
            )
        except SystemExit:
            beat(f"OPTIONAL train_swin_seed{seed} failed — 忽略（保险批次）")
            break
    beat("=== optional seed replicates finished ===")


if __name__ == "__main__":
    main()
