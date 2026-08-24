"""构造可直接供现有终稿构建链消费的隔离 composite results view。"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from strict_oof.protocol import (
    BACKBONES,
    DELIVERY_LEGACY_FILES,
    TRAIN_MODELS,
    atomic_write_json,
    file_sha256,
    paths,
)
from strict_oof.train import artifact_paths


def _copy_verified(source: Path, target: Path, expected_sha256: str) -> None:
    """复制合同绑定文件，并在复制前后验证哈希。"""
    if not source.is_file() or file_sha256(source) != expected_sha256:
        raise RuntimeError(f"交付源与合同不匹配: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if file_sha256(target) != expected_sha256:
        raise RuntimeError(f"交付复制后哈希不匹配: {target}")


def _gpu_summary(samples: list[float]) -> dict[str, float | int]:
    values = np.asarray(samples or [0.0], dtype=float)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "n": int(len(values)),
    }


def _main_training_summaries(
    staging: Path, contract: dict[str, Any], analysis: dict[str, Any]
) -> None:
    """生成当前构建器可读的严格训练聚合事实，不复用旧主训日志。"""
    for model in TRAIN_MODELS:
        outer_logs = []
        for fold in range(5):
            _checkpoint, _resume, log_path, _lock = artifact_paths(
                "outer", model, fold=fold
            )
            outer_logs.append(json.loads(log_path.read_text(encoding="utf-8")))
        _checkpoint, _resume, terminal_path, _lock = artifact_paths("terminal", model)
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        samples = [
            float(value)
            for record in [*outer_logs, terminal]
            for value in record.get("gpu_utilization_samples", [])
        ]
        if model in BACKBONES:
            metacal_f1 = analysis["best_single_selection"][
                "metacal_macro_f1_descriptive"
            ][model]
        else:
            metacal_f1 = analysis["same_protocol_resnet50"]["metacal"]["macro_f1"]
        summary = {
            "tag": model,
            "model_name": terminal["model_name"],
            "status": "complete",
            "contract_sha256": contract["contract_sha256"],
            "summary_type": "strict_outer5_plus_terminal_aggregate",
            "max_epochs": 40,
            "stopped_epoch": terminal["stopped_epoch"],
            "best_epoch": terminal["best_epoch"],
            "terminal_fixed_epoch": terminal["best_epoch"],
            "outer_best_epochs": [row["best_epoch"] for row in outer_logs],
            "outer_stopped_epochs": [row["stopped_epoch"] for row in outer_logs],
            "best_metacal_macro_f1": float(metacal_f1),
            "independent_metacal_macro_f1": float(metacal_f1),
            "metacal_metric_role": "independent evaluation; not epoch selection",
            "total_seconds": float(
                sum(row["total_seconds"] for row in [*outer_logs, terminal])
            ),
            "gpu_utilization": _gpu_summary(samples),
            "hparams": terminal["hparams"],
            "scheduler_horizon_epochs": 40,
            "outer_early_stopping_partition": "patient-grouped inner validation",
            "terminal_early_stopping": False,
            "terminal_epoch_rule": "median_of_five_outer_best_epochs",
            "test_labels_used_for_training_or_selection": False,
        }
        atomic_write_json(staging / f"train_log_{model}.json", summary)


def _ablation_training_logs(
    staging: Path, contract: dict[str, Any], extended: dict[str, Any]
) -> None:
    """发布旧六臂合同绑定日志与新三臂严格日志。"""
    p = paths()
    for name, expected in contract["legacy_ablation_train_log_sha256"].items():
        _copy_verified(p.exp / "results" / name, staging / name, expected)
    backbone = contract["additional_ablation_backbone"]
    for arm in ("cb_only", "mixup_only", "cutmix_only"):
        _checkpoint, _resume, log_path, _lock = artifact_paths(
            "ablation", backbone, arm=arm
        )
        record = json.loads(log_path.read_text(encoding="utf-8"))
        values = [float(value) for value in record.get("gpu_utilization_samples", [])]
        compatible = dict(record)
        compatible.update(
            {
                "tag": f"{backbone}_abl_{arm}",
                "best_metacal_macro_f1": record["best_monitor_macro_f1"],
                "gpu_utilization": _gpu_summary(values),
                "schema_note": "strict isolated additional ablation arm",
                "minority_overcompensation_index": extended["arms"][arm][
                    "minority_overcompensation_index"
                ],
            }
        )
        atomic_write_json(staging / f"train_log_{backbone}_abl_{arm}.json", compatible)


def _payload_hashes(staging: Path) -> dict[str, str]:
    """计算 composite payload 哈希，不包含其自身清单。"""
    result = {}
    for target in sorted(staging.rglob("*"), key=lambda item: item.as_posix()):
        if target.is_file() and target.name != "payload_manifest.json":
            result[target.relative_to(staging).as_posix()] = file_sha256(target)
    return result


def _expected_delivery_files(
    delivery: Path, contract: dict[str, Any], *, include_core: bool
) -> set[str]:
    """返回发布视图的封闭白名单；manifest不能自我授权额外文件。"""
    expected = {
        *DELIVERY_LEGACY_FILES,
        "image_quality.json",
        "analysis.json",
        "ablation_analysis.json",
        "strict_validation_report.json",
        "frozen_baseline_provenance.json",
        "STRICT_DELIVERY.json",
        *(f"train_log_{model}.json" for model in TRAIN_MODELS),
        *contract["legacy_ablation_train_log_sha256"],
        *(
            f"train_log_{contract['additional_ablation_backbone']}_abl_{arm}.json"
            for arm in ("cb_only", "mixup_only", "cutmix_only")
        ),
    }
    try:
        cases = json.loads((delivery / "gradcam_index.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        cases = []
    expected.add("gradcam_index.json")
    for row in cases if isinstance(cases, list) else []:
        if isinstance(row, dict) and isinstance(row.get("case_id"), str):
            expected.add(f"gradcam/{row['case_id']}.png")
    if include_core:
        expected.update(
            {
                "core_validation.json",
                "markers/core_validation.done",
                "markers/ALL.done",
            }
        )
    return expected


def _existing_valid(delivery: Path, contract: dict[str, Any]) -> bool:
    marker = delivery / "STRICT_DELIVERY.json"
    manifest = delivery / "payload_manifest.json"
    if not marker.is_file() or not manifest.is_file():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        metadata.get("contract_sha256") != contract["contract_sha256"]
        or payload.get("contract_sha256") != contract["contract_sha256"]
    ):
        return False
    files = payload.get("files")
    if not isinstance(files, dict):
        return False
    actual = {
        target.relative_to(delivery).as_posix()
        for target in delivery.rglob("*")
        if target.is_file() and target.name != "payload_manifest.json"
    }
    expected = _expected_delivery_files(
        delivery, contract, include_core="core_validation.json" in actual
    )
    if actual != set(files) or actual != expected:
        return False
    for relative, expected in files.items():
        target = delivery / relative
        if not target.is_file() or file_sha256(target) != expected:
            return False
    return True


def refresh_payload_manifest(
    delivery: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    """core三门写入后重新冻结精确文件集合，拒绝任何额外文件。"""
    payload = {
        "contract_sha256": contract["contract_sha256"],
        "files": _payload_hashes(delivery),
    }
    atomic_write_json(delivery / "payload_manifest.json", payload)
    if not _existing_valid(delivery, contract):
        raise RuntimeError("refreshed delivery payload manifest failed self-check")
    return payload


def build_delivery(contract: dict[str, Any]) -> Path:
    """原子构造完整 results view；不写旧 ``results`` 产物。"""
    p = paths()
    if _existing_valid(p.delivery, contract):
        return p.delivery
    staging = p.out / f".delivery.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    legacy_root = p.exp / "results"
    try:
        for name in DELIVERY_LEGACY_FILES:
            _copy_verified(
                legacy_root / name,
                staging / name,
                contract["legacy_delivery_sha256"][name],
            )
        _copy_verified(
            p.out / "image_quality.json",
            staging / "image_quality.json",
            contract["strict_image_quality_sha256"],
        )
        analysis_path = p.out / "analysis.json"
        extended_path = p.out / "extended_ablation_analysis.json"
        validation_path = p.validation_report
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        extended = json.loads(extended_path.read_text(encoding="utf-8"))
        shutil.copy2(analysis_path, staging / "analysis.json")
        ablation = dict(extended["arms"])
        ablation["_backbone"] = extended["backbone"]
        atomic_write_json(staging / "ablation_analysis.json", ablation)
        shutil.copy2(validation_path, staging / "strict_validation_report.json")

        gradcam_index = json.loads(
            (p.out / "gradcam_index.json").read_text(encoding="utf-8")
        )
        atomic_write_json(staging / "gradcam_index.json", gradcam_index["cases"])
        for row in gradcam_index["cases"]:
            source = p.out / "gradcam" / f"{row['case_id']}.png"
            _copy_verified(
                source,
                staging / "gradcam" / source.name,
                row["png_sha256"],
            )

        _main_training_summaries(staging, contract, analysis)
        _ablation_training_logs(staging, contract, extended)
        atomic_write_json(
            staging / "frozen_baseline_provenance.json",
            {
                "source_sha256": contract["legacy_delivery_sha256"][
                    "frozen_baseline.json"
                ],
                "source_split": "same frozen Train/Meta-Cal/Test assignment",
                "role": "frozen-feature classical comparator",
                "same_protocol_training_recipe": False,
                "note": (
                    "This legacy comparator is contract-bound and retained only "
                    "for its stated frozen-feature role; it is not described as "
                    "a strict-OOF end-to-end backbone run."
                ),
            },
        )
        atomic_write_json(
            staging / "STRICT_DELIVERY.json",
            {
                "status": "complete_payload",
                "protocol_id": "strict_oof_v1",
                "contract_sha256": contract["contract_sha256"],
                "strict_validation_report_sha256": file_sha256(validation_path),
                "legacy_results_overwritten": False,
                "analysis_source": "strict Train-OOF / Meta-Cal / terminal-Test",
                "ablation_arms": sorted(extended["arms"]),
                "gradcam_source": "strict terminal ConvNeXt-B TTA0",
            },
        )
        payload = {
            "contract_sha256": contract["contract_sha256"],
            "files": _payload_hashes(staging),
        }
        atomic_write_json(staging / "payload_manifest.json", payload)
        if p.delivery.exists():
            quarantine = p.out / "delivery_quarantine"
            quarantine.mkdir(parents=True, exist_ok=True)
            p.delivery.replace(quarantine / f"delivery-{uuid.uuid4().hex}")
        staging.replace(p.delivery)
    except Exception:
        # 失败临时目录保留供根因审计，不发布指针。
        raise
    return p.delivery
