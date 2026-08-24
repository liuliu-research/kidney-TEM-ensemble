"""严格 OOF 无人值守编排器；步数以 workflow 常量为唯一真值。"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from strict_oof.protocol import (
    atomic_write_json,
    build_contract,
    ensure_runtime_dirs,
    file_sha256,
    load_contract,
    paths,
)
from strict_oof.predict import prediction_paths
from strict_oof.train import artifact_paths
from strict_oof.train import exclusive_lock
from strict_oof.validate import _prediction_step_parts, _train_step_parts
from strict_oof.workflow import (
    EXPECTED_TOTAL_STEPS,
    EXPECTED_TRAINING_RUNS,
    Step,
    build_plan,
)

PYTHON = sys.executable
EXP_DIR = Path(__file__).resolve().parents[1]
GPU_GATE_EVER_PASSED = False


def _module_command(args: tuple[str, ...]) -> list[str]:
    """把计划中的可读脚本路径映射为可移植的 ``-m`` 入口。"""
    first, *rest = args
    if first.startswith("strict_oof/") and first.endswith(".py"):
        module = first.removesuffix(".py").replace("/", ".")
        return [PYTHON, "-m", module, *rest]
    return [PYTHON, *args]


def _validation_command(step_name: str) -> list[str]:
    return [PYTHON, "-m", "strict_oof.validate", "--step", step_name]


def _expected_marker_contract(step: Step) -> str:
    if step.name == "image_quality_v2":
        return "precontract:image_quality_v2"
    return load_contract(fresh=False)["contract_sha256"]


def _step_output_paths(step: Step) -> list[Path]:
    """返回需写入 marker 哈希契约的当步完成产物。"""
    p = paths()
    targets: list[Path] = []
    if step.name == "image_quality_v2":
        return [p.out / "image_quality.json"]
    if step.name == "prepare_contract":
        return [p.contract]
    train_parts = _train_step_parts(step.name)
    if train_parts:
        stage, model, fold, arm = train_parts
        if stage == "ablation":
            model = load_contract(fresh=False)["additional_ablation_backbone"]
        checkpoint, _resume, log, _lock = artifact_paths(
            stage, model, fold=fold, arm=arm
        )
        return [checkpoint, log]
    prediction_parts = _prediction_step_parts(step.name)
    if prediction_parts:
        stage, model, subset, fold, arm = prediction_parts
        if stage == "ablation":
            model = load_contract(fresh=False)["additional_ablation_backbone"]
        bundle, _shards, meta, _lock = prediction_paths(
            stage, model, subset, fold=fold, arm=arm
        )
        return [bundle, meta]
    targets.extend(
        {
            "strict_analysis": [p.out / "analysis.json"],
            "strict_gradcam": [p.out / "gradcam_index.json", p.out / "gradcam"],
            "extended_ablation_analysis": [p.out / "extended_ablation_analysis.json"],
            "validate_strict_oof": [p.validation_report],
            "finalize_strict_oof": [
                p.ready,
                p.delivery,
                p.exp / "results" / "strict_oof_ACTIVE.json",
            ],
        }.get(step.name, [])
    )
    return targets


def _step_output_hashes(step: Step) -> dict[str, str]:
    """对单文件或目录内所有文件计算稳定哈希。"""
    p = paths()
    hashes: dict[str, str] = {}
    for target in _step_output_paths(step):
        files = (
            sorted(target.rglob("*"), key=lambda item: item.as_posix())
            if target.is_dir()
            else [target]
        )
        for current in files:
            if current.is_file():
                try:
                    key = current.resolve().relative_to(p.exp.resolve()).as_posix()
                except ValueError:
                    key = str(current.resolve())
                hashes[key] = file_sha256(current)
    return hashes


def _status_payload(current_step: str | None, message: str) -> dict:
    p = paths()
    markers = list(p.markers.glob("*.done")) if p.markers.exists() else []
    failures = list(p.markers.glob("*.FAILED")) if p.markers.exists() else []
    return {
        "updated_at": datetime.now().isoformat(),
        "current_step": current_step,
        "message": message,
        "completed_steps": len(markers),
        "failed_markers": sorted(path.name for path in failures),
        "ready": p.ready.is_file(),
        "total_steps": len(build_plan()),
    }


def beat(message: str, current_step: str | None = None) -> None:
    """同时更新人类日志和机器状态。"""
    p = paths()
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}"
    print(line, flush=True)
    p.logs.mkdir(parents=True, exist_ok=True)
    with (p.logs / "status.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    atomic_write_json(
        p.out / "workflow_status.json", _status_payload(current_step, message)
    )


def write_plan(contract_hash: str | None) -> None:
    """发布与执行顺序同源的计划 JSON。"""
    plan = build_plan()
    payload = {
        "protocol": "strict_oof_v1",
        "contract_sha256": contract_hash,
        "total_steps": len(plan),
        "gpu_training_runs": EXPECTED_TRAINING_RUNS,
        "outer_training_runs": 30,
        "terminal_training_runs": 6,
        "additional_ablation_training_runs": 3,
        "estimated_single_gpu_wall_time": (
            "approximately 6-7 days from legacy epoch timings; runtime is "
            "descriptive and the live logs are authoritative"
        ),
        "legacy_results_overwritten": False,
        "steps": [step.as_dict() for step in plan],
    }
    atomic_write_json(paths().out / "training_plan.json", payload)


def _is_allowlisted_kidney_backend(
    process_name: str,
    used_gpu_memory_mib: str,
    cgroup: str,
    cmdline: str,
) -> bool:
    """只豁免 systemd 管理的低显存 kidney-backend uvicorn API。"""
    try:
        memory = float(used_gpu_memory_mib.strip())
    except ValueError:
        return False
    basename = Path(process_name.strip()).name.lower()
    process_ok = re.fullmatch(r"(?:uvicorn|python|python3|python3\.\d+)", basename)
    command = cmdline.replace("\x00", " ").lower()
    cgroup_paths = {
        parts[2]
        for line in cgroup.splitlines()
        if len(parts := line.split(":", 2)) == 3
    }
    return bool(
        0 <= memory <= 512
        and process_ok
        and "/system.slice/kidney-backend.service" in cgroup_paths
        and "uvicorn" in command
        and ("kidney" in command or "app.main" in command)
    )


def _gpu_snapshot() -> tuple[float, list[str]]:
    utilization = 100.0
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
        utilization = float(result.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    foreign: list[str] = []
    try:
        compute_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if compute_result.returncode != 0:
            foreign.append("gpu compute-app query failed")
        compute_output = compute_result.stdout
        own_pids = {str(Path("/proc/self").resolve().name), str(os.getpid())}
        for line in compute_output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                foreign.append(f"gpu-compute-unparseable:{line.strip()}")
                continue
            pid, process_name, used_memory = parts
            if not pid or pid in own_pids:
                continue
            try:
                cgroup = Path(f"/proc/{pid}/cgroup").read_text(
                    encoding="utf-8", errors="replace"
                )
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(
                    "utf-8", errors="replace"
                )
            except OSError:
                foreign.append(f"gpu-compute-unverifiable:{line.strip()}")
                continue
            if _is_allowlisted_kidney_backend(
                process_name, used_memory, cgroup, cmdline
            ):
                continue
            foreign.append(f"gpu-compute:{line.strip()}")
        process_result = subprocess.run(
            ["ps", "-eo", "pid,args"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if process_result.returncode != 0:
            foreign.append("process scan failed")
        process_output = process_result.stdout
        patterns = (
            "train_image.py",
            "train_text.py",
            "train_kg.py",
            "train_fusion.py",
            "run_v2_master_parallel.sh",
            "baselines/",
            # 旧项目3主线与分析也必须显式识别。
            "train_backbone.py",
            "revision_experiments/orchestrator.py",
            "fusion_analysis.py",
            "ablation_analysis.py",
            "gradcam.py",
            "profile_io.py",
            "kgmm_milestone_launch.sh",
            "kgmm_train_guard.sh",
            "run_full_retrain_20260526.sh",
            "multimodal_train_v3.py",
            "train_loho_v3.py",
            "autostart_retrain.sh",
        )
        for line in process_output.splitlines():
            if "strict_oof/" in line:
                continue
            if any(pattern in line for pattern in patterns):
                foreign.append(line.strip())
    except (OSError, subprocess.SubprocessError):
        foreign.append("process scan failed")
    return utilization, foreign


def wait_for_gpu() -> None:
    """外部作业或高利用率存在时始终只等待。"""
    global GPU_GATE_EVER_PASSED
    utilization, foreign = _gpu_snapshot()
    if GPU_GATE_EVER_PASSED and utilization < 15 and not foreign:
        beat(f"GPU gate revalidated (util={utilization:.1f}%)")
        return
    streak = 0
    while streak < 3:
        utilization, foreign = _gpu_snapshot()
        if utilization < 15 and not foreign:
            streak += 1
            beat(f"GPU idle gate {streak}/3 (util={utilization:.1f}%)")
        else:
            streak = 0
            beat(f"WAIT GPU util={utilization:.1f}% foreign={foreign[:3]}")
        if streak < 3:
            time.sleep(120)
    GPU_GATE_EVER_PASSED = True


def _marker_valid(step: Step) -> bool:
    marker = paths().markers / f"{step.name}.done"
    if not marker.is_file():
        return False
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        record = {}
    current_hashes = _step_output_hashes(step)
    marker_metadata_ok = (
        record.get("step") == step.name
        and record.get("contract_sha256") == _expected_marker_contract(step)
        and record.get("command") == _module_command(step.args)
        and record.get("artifact_sha256") == current_hashes
        and bool(current_hashes)
    )
    result = subprocess.run(
        _validation_command(step.name),
        cwd=EXP_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode == 0 and marker_metadata_ok:
        return True
    quarantine = paths().out / "marker_quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    marker.replace(quarantine / f"{marker.name}.{datetime.now():%Y%m%d_%H%M%S}.invalid")
    if result.returncode != 0 or (
        record.get("artifact_sha256")
        and record.get("artifact_sha256") != current_hashes
    ):
        _quarantine_step_outputs(step)
    beat(f"QUARANTINE invalid marker {step.name}", step.name)
    return False


def _publish_marker(step: Step) -> None:
    contract_hash = _expected_marker_contract(step)
    artifact_hashes = _step_output_hashes(step)
    if not artifact_hashes:
        raise RuntimeError(f"refuse marker without hashed artifacts: {step.name}")
    atomic_write_json(
        paths().markers / f"{step.name}.done",
        {
            "step": step.name,
            "completed_at": datetime.now().isoformat(),
            "contract_sha256": contract_hash,
            "command": _module_command(step.args),
            "artifact_sha256": artifact_hashes,
        },
    )


def _quarantine_step_outputs(step: Step) -> None:
    """只隔离当步的主产物，保留 epoch/TTA shards 以供精确恢复。"""
    p = paths()
    quarantine = p.out / "artifact_quarantine" / step.name / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    targets: list[Path] = []
    if step.name == "image_quality_v2":
        targets.append(p.out / "image_quality.json")
    if step.name == "prepare_contract":
        targets.append(p.contract)
    train_parts = _train_step_parts(step.name)
    if train_parts:
        stage, model, fold, arm = train_parts
        if stage == "ablation":
            model = load_contract(fresh=False)["additional_ablation_backbone"]
        checkpoint, _resume, log, _lock = artifact_paths(
            stage, model, fold=fold, arm=arm
        )
        targets.extend((checkpoint, log))
    prediction_parts = _prediction_step_parts(step.name)
    if prediction_parts:
        stage, model, subset, fold, arm = prediction_parts
        if stage == "ablation":
            model = load_contract(fresh=False)["additional_ablation_backbone"]
        bundle, _shards, meta, _lock = prediction_paths(
            stage, model, subset, fold=fold, arm=arm
        )
        targets.extend((bundle, meta))
    targets.extend(
        {
            "strict_analysis": [p.out / "analysis.json"],
            "strict_gradcam": [p.out / "gradcam_index.json", p.out / "gradcam"],
            "extended_ablation_analysis": [
                p.out / "extended_ablation_analysis.json"
            ],
            "validate_strict_oof": [p.validation_report],
            "finalize_strict_oof": [p.ready, p.delivery],
        }.get(step.name, [])
    )
    if step.name == "finalize_strict_oof":
        targets.append(p.exp / "results" / "strict_oof_ACTIVE.json")
    existing = [target for target in targets if target.exists()]
    if not existing:
        return
    quarantine.mkdir(parents=True, exist_ok=True)
    for target in existing:
        destination = quarantine / target.name
        if destination.exists():
            destination = quarantine / f"{target.name}-{len(list(quarantine.iterdir()))}"
        target.replace(destination)
    beat(f"QUARANTINE invalid outputs for {step.name}: {len(existing)}", step.name)


def run_step(step: Step) -> None:
    """有界重试；每次训练失败都从 epoch 断点恢复。"""
    if _marker_valid(step):
        beat(f"SKIP {step.name} (validated marker)")
        return
    # marker 可能因断电尚未发布，但主产物已存在且损坏。
    # 先隔离再调生产者，避免浅层 existing-complete 判定导致死循环。
    candidate_exists = any(target.exists() for target in _step_output_paths(step))
    if candidate_exists:
        validation = subprocess.run(
            _validation_command(step.name),
            cwd=EXP_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if validation.returncode != 0:
            _quarantine_step_outputs(step)
    if step.gpu:
        wait_for_gpu()
    log_path = paths().logs / f"{step.name}.log"
    for attempt in range(step.retries + 1):
        beat(f"START {step.name} attempt={attempt + 1}", step.name)
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                _module_command(step.args),
                cwd=EXP_DIR,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode == 0:
            with log_path.open("a", encoding="utf-8") as validation_log:
                validation = subprocess.run(
                    _validation_command(step.name),
                    cwd=EXP_DIR,
                    stdout=validation_log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if validation.returncode == 0:
                _publish_marker(step)
                (paths().markers / f"{step.name}.FAILED").unlink(missing_ok=True)
                beat(f"DONE {step.name}")
                return
            _quarantine_step_outputs(step)
            result = validation
        beat(f"FAIL {step.name} rc={result.returncode}", step.name)
        if attempt < step.retries:
            time.sleep(60)
    atomic_write_json(
        paths().markers / f"{step.name}.FAILED",
        {
            "step": step.name,
            "failed_at": datetime.now().isoformat(),
            "attempts_in_orchestrator_turn": step.retries + 1,
        },
    )
    raise RuntimeError(f"persistent step failure: {step.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        try:
            candidate = build_contract()
            contract_summary: dict[str, object] = {
                key: candidate[key]
                for key in (
                    "contract_sha256",
                    "train_images",
                    "metacal_images",
                    "test_images",
                    "additional_ablation_backbone",
                )
            }
            preflight_ready = True
            blockers: list[str] = []
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
            contract_summary = {"status": "not_yet_constructible"}
            preflight_ready = False
            blockers = [str(error)]
        print(
            json.dumps(
                {
                    "preflight_ready": preflight_ready,
                    "preflight_blockers": blockers,
                    "contract_summary": contract_summary,
                    "plan": [step.as_dict() for step in build_plan()],
                    "total_steps": EXPECTED_TOTAL_STEPS,
                    "gpu_training_runs": EXPECTED_TRAINING_RUNS,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    p = paths()
    ensure_runtime_dirs(p)
    with exclusive_lock(p.big / "locks" / "orchestrator.lock"):
        beat("=== strict OOF orchestrator start ===")
        for step in build_plan():
            if step.name == "prepare_contract":
                run_step(step)
                write_plan(load_contract(fresh=False)["contract_sha256"])
            else:
                run_step(step)
        beat("=== STRICT OOF READY ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
