"""严格 OOF 补训的整点飞书进度报告。"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

EXPECTED_TRAINING_RUNS = 39
EPOCH_RE = re.compile(r"\bep(\d+)\b.*\((\d+)s\)\s*$")
EXP_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_MODULE_ROOT))


@dataclass(frozen=True)
class ReportPaths:
    """播报器所需路径。"""

    exp: Path
    out: Path
    markers: Path
    logs: Path
    plan: Path
    status: Path
    ready: Path
    push: Path


def report_paths(exp_dir: Path | None = None) -> ReportPaths:
    """解析生产或测试路径。"""
    exp = Path(
        exp_dir or os.environ.get("SR_EXP_DIR") or Path(__file__).resolve().parents[1]
    ).resolve()
    out = exp / "results" / "strict_oof_v1"
    return ReportPaths(
        exp=exp,
        out=out,
        markers=out / "markers",
        logs=exp / "logs" / "strict_oof_v1",
        plan=out / "training_plan.json",
        status=out / "workflow_status.json",
        ready=out / "READY.json",
        push=exp / "push.sh",
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_plan(paths: ReportPaths) -> dict[str, Any]:
    """优先读已发布计划；首步尚未发布时从同源代码构造。"""
    plan = _load_json(paths.plan)
    if isinstance(plan.get("steps"), list):
        return plan
    from strict_oof.workflow import build_plan

    steps = [step.as_dict() for step in build_plan()]
    return {
        "steps": steps,
        "total_steps": len(steps),
        "gpu_training_runs": EXPECTED_TRAINING_RUNS,
        "source": "workflow.py fallback before training_plan.json publication",
    }


def _plan_counts(plan: dict[str, Any]) -> tuple[int, int, int]:
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return 0, EXPECTED_TRAINING_RUNS, 0
    training = 0
    predictions = 0
    for row in steps:
        if not isinstance(row, dict):
            continue
        args = row.get("args")
        if not isinstance(args, list) or not args:
            continue
        if args[0] == "strict_oof/train.py":
            training += 1
        elif args[0] == "strict_oof/predict.py":
            predictions += 1
    if training != EXPECTED_TRAINING_RUNS:
        raise RuntimeError(
            f"训练计划必须恰好 {EXPECTED_TRAINING_RUNS} 个完整训练 run，实际 {training}"
        )
    return len(steps), training, predictions


def _done_names(paths: ReportPaths) -> set[str]:
    if not paths.markers.is_dir():
        return set()
    return {path.stem for path in paths.markers.glob("*.done")}


def _first_pending_step(plan: dict[str, Any], done: set[str]) -> str | None:
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return None
    for row in steps:
        if isinstance(row, dict) and str(row.get("name", "")) not in done:
            return str(row["name"])
    return None


def _current_step(paths: ReportPaths, plan: dict[str, Any], done: set[str]) -> str:
    status = _load_json(paths.status)
    current = status.get("current_step")
    plan_names = {
        str(row.get("name")) for row in plan.get("steps", []) if isinstance(row, dict)
    }
    if isinstance(current, str) and current in plan_names and current not in done:
        return current
    return _first_pending_step(plan, done) or "收尾深验收"


def _stage_text(step: str) -> str:
    match = re.fullmatch(r"outer_train_(.+)_fold([0-4])", step)
    if match:
        return f"Train内严格OOF｜{match.group(1)} fold{int(match.group(2)) + 1}/5"
    match = re.fullmatch(r"terminal_train_(.+)", step)
    if match:
        return f"完整Train固定轮次重训｜{match.group(1)}"
    match = re.fullmatch(r"ablation_train_(.+)", step)
    if match:
        return f"独立消融补训｜{match.group(1)}"
    if "pred" in step:
        return "严格OOF/终局推理"
    if "gradcam" in step:
        return "终局组件级Grad-CAM"
    if "image_quality" in step:
        return "真实输入链图像保真"
    if "analysis" in step:
        return "统计分析与九臂消融汇总"
    if "validate" in step or "finalize" in step or "delivery" in step:
        return "终局深验收与交付视图"
    return "协议准备/工作流收尾"


def _terminal_epoch(paths: ReportPaths, step: str) -> int | None:
    match = re.fullmatch(r"terminal_train_(.+)", step)
    if not match:
        return None
    values: list[int] = []
    for fold in range(5):
        row = _load_json(
            paths.out / "train" / f"outer_fold{fold}_{match.group(1)}.json"
        )
        try:
            values.append(int(row["best_epoch"]))
        except (KeyError, TypeError, ValueError):
            return None
    return int(statistics.median(values))


def _epoch_text(paths: ReportPaths, step: str) -> tuple[str, str, int | None]:
    step_log = paths.logs / f"{step}.log"
    if not step_log.is_file():
        return "步骤刚启动，日志尚未生成", "首轮计时中", None
    rows: list[tuple[int, int]] = []
    try:
        for line in step_log.read_text(encoding="utf-8", errors="replace").splitlines():
            match = EPOCH_RE.search(line)
            if match:
                rows.append((int(match.group(1)), int(match.group(2))))
    except OSError:
        rows = []
    training = (
        step.startswith("outer_train_")
        or step.startswith("terminal_train_")
        or step.startswith("ablation_train_")
    )
    if not training:
        return "该步骤无 epoch 概念", "通常数分钟至数小时", None
    if not rows:
        return "数据装载/首轮进行中", "首轮计时中", None
    epoch = rows[-1][0]
    avg = round(sum(value for _, value in rows[-3:]) / len(rows[-3:]))
    if step.startswith("ablation_train_"):
        maximum = 30
    elif step.startswith("outer_train_"):
        maximum = 40
    else:
        maximum = _terminal_epoch(paths, step) or epoch
    minutes = max(maximum - epoch, 0) * avg // 60
    progress = f"已完成 ep{epoch}/{maximum}｜每轮 {avg // 60}分{avg % 60}秒"
    eta = f"本模型最多再需 {minutes // 60}h{minutes % 60}m（触发早停会更快）"
    if step.startswith("terminal_train_"):
        eta = f"本模型固定轮次最多再需 {minutes // 60}h{minutes % 60}m"
    return progress, eta, epoch


def _gpu_utilization() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return f"{int(float(result.stdout.strip().splitlines()[0]))} %"
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return "不可读"


def _orchestrator_alive(paths: ReportPaths) -> bool:
    proc = Path("/proc")
    if not proc.is_dir():
        return True
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
            cwd = (entry / "cwd").resolve()
        except OSError:
            continue
        if (
            b"strict_oof/orchestrator.py" in command
            or b"strict_oof.orchestrator" in command
        ) and cwd == paths.exp:
            return True
    return False


def _health(paths: ReportPaths, step: str) -> tuple[str, str]:
    failed = sorted(path.stem for path in paths.markers.glob("*.FAILED"))
    step_log = paths.logs / f"{step}.log"
    age = None
    if step_log.is_file():
        age = max(0, int((datetime.now().timestamp() - step_log.stat().st_mtime) / 60))
    elif paths.status.is_file():
        age = max(
            0,
            int((datetime.now().timestamp() - paths.status.stat().st_mtime) / 60),
        )
    suffix = f"日志 {age} 分钟前更新" if age is not None else "当前步骤尚无日志文件"
    if failed:
        return (
            f"🛑 FAILED：{'、'.join(failed)}｜GPU {_gpu_utilization()}｜{suffix}",
            "red",
        )
    if not _orchestrator_alive(paths) and not paths.ready.is_file():
        return (
            f"🛑 编排器不存在（看门狗将自动拉起）｜GPU {_gpu_utilization()}｜{suffix}",
            "red",
        )
    if age is not None:
        analysis_step = any(
            token in step
            for token in ("analysis", "validate", "finalize", "delivery", "gradcam")
        )
        if analysis_step and age > 720:
            return (
                f"🛑 分析步骤日志已停更 {age} 分钟｜GPU {_gpu_utilization()}",
                "red",
            )
        if analysis_step and age > 120:
            return (
                f"⚠️ 分析步骤日志已停更 {age} 分钟（12小时硬阈值）｜"
                f"GPU {_gpu_utilization()}",
                "orange",
            )
        if not analysis_step and age > 120:
            return (
                f"🛑 当前步骤日志已停更 {age} 分钟，已超过120分钟阈值｜"
                f"GPU {_gpu_utilization()}",
                "red",
            )
    return f"GPU {_gpu_utilization()}｜{suffix}", "blue"


def build_report(paths: ReportPaths) -> dict[str, Any]:
    """构建机器可测的飞书卡片内容。"""
    plan = _load_plan(paths)
    total_steps, total_training, total_predictions = _plan_counts(plan)
    done = _done_names(paths)
    training_names = {
        str(row["name"])
        for row in plan.get("steps", [])
        if isinstance(row, dict)
        and isinstance(row.get("args"), list)
        and row["args"]
        and row["args"][0] == "strict_oof/train.py"
    }
    prediction_names = {
        str(row["name"])
        for row in plan.get("steps", [])
        if isinstance(row, dict)
        and isinstance(row.get("args"), list)
        and row["args"]
        and row["args"][0] == "strict_oof/predict.py"
    }
    plan_names = {
        str(row.get("name")) for row in plan.get("steps", []) if isinstance(row, dict)
    }
    ready = _load_json(paths.ready)
    completion = (
        ready.get("status") == "ready"
        and "finalize_strict_oof" in done
        and not list(paths.markers.glob("*.FAILED"))
        and plan_names <= done
        and len(done & training_names) == total_training
        and len(done & prediction_names) == total_predictions
    )
    if completion:
        return {
            "complete": True,
            "title": "✅项目3严格补训全部完成",
            "body": (
                f"完整训练 {len(done & training_names)}/{total_training} · "
                f"推理 {len(done & prediction_names)}/{total_predictions} · "
                f"总步骤 {len(done & plan_names)}/{total_steps}；"
                "严格OOF、九臂消融、终局分析与核心产物深验收全部通过。"
                "项目3本轮补训真正完工，停播。"
            ),
            "color": "green",
        }
    step = _current_step(paths, plan, done)
    progress, eta, _epoch = _epoch_text(paths, step)
    health, color = _health(paths, step)
    return {
        "complete": False,
        "title": f"📊严格补训进度 {datetime.now():%m-%d %H:%M}",
        "body": (
            f"阶段 {_stage_text(step)}｜当前：{step}\n"
            f"{progress}\n{eta}\n\n"
            f"总步骤：{len(done & plan_names)}/{total_steps}\n"
            f"完整训练：{len(done & training_names)}/{total_training}｜"
            f"推理：{len(done & prediction_names)}/{total_predictions}\n"
            f"{health}"
        ),
        "color": color,
    }


def _send(paths: ReportPaths, report: dict[str, Any]) -> int:
    marker = paths.markers / "STRICT_TRULY_DONE.notified"
    hour_marker = paths.logs / f"hourly_sent_{datetime.now():%Y%m%d_%H}"
    if report["complete"] and marker.is_file():
        return 0
    if hour_marker.is_file():
        return 0
    command = [
        "bash",
        str(paths.push),
        str(report["title"]),
        str(report["body"]),
        str(report["color"]),
    ]
    returncode = 1
    for attempt in range(3):
        try:
            result = subprocess.run(
                command,
                cwd=paths.exp,
                timeout=60,
                check=False,
            )
            returncode = result.returncode
        except (OSError, subprocess.TimeoutExpired):
            returncode = 1
        if returncode == 0:
            break
        if attempt < 2:
            time.sleep(5)
    if returncode != 0:
        return returncode
    paths.logs.mkdir(parents=True, exist_ok=True)
    hour_tmp = hour_marker.with_suffix(".tmp")
    hour_tmp.write_text(datetime.now().isoformat() + "\n", encoding="utf-8")
    hour_tmp.replace(hour_marker)
    if report["complete"]:
        marker.parent.mkdir(parents=True, exist_ok=True)
        temporary = marker.with_suffix(marker.suffix + ".tmp")
        temporary.write_text(datetime.now().isoformat() + "\n", encoding="utf-8")
        temporary.replace(marker)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--exp-dir", type=Path)
    args = parser.parse_args()
    paths = report_paths(args.exp_dir)
    report = build_report(paths)
    if args.send:
        return _send(paths, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
