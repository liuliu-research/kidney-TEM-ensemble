"""SR 大修实验 45 个核心步骤的 fail-closed 完整性验收。

本脚本只读实验产物，不执行训练或分析。它不把 ``*.done`` 当作成功的
充分条件：marker、JSON、权重、预测 NPZ 与终局图像必须同时通过检查。

退出码：0=全部通过；1=任一检查失败（报告仍尽力写入 ``--json-out``）。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

MAX_CACHE_QC_EXCLUSIONS = 3
Image.MAX_IMAGE_PIXELS = None


def _reject_json_constant(token: str) -> None:
    """拒绝 Python JSON 解码器默认容忍的 NaN/Infinity。"""
    raise ValueError(f"非标准 JSON 数值: {token}")


@dataclass(frozen=True)
class Expectations:
    """从 ``config.py`` 静态读取的稳定命名与类别定义。"""

    classes: tuple[str, ...]
    backbones: tuple[str, ...]
    baseline: tuple[str, ...]
    ablations: tuple[str, ...]
    tta_count: int

    @property
    def models(self) -> tuple[str, ...]:
        return self.backbones + self.baseline


def _literal_assignments(config_path: Path) -> dict[str, Any]:
    """不执行 ``config.py``，只读取所需的字面量赋值。"""
    tree = ast.parse(config_path.read_text(encoding="utf-8"), config_path.name)
    wanted = {
        "CLASS_FOLDERS",
        "BACKBONES",
        "BASELINE_BACKBONE",
        "ABLATIONS",
        "TTA_SWEEP",
    }
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            values[target.id] = ast.literal_eval(node.value)
    missing = sorted(wanted - values.keys())
    if missing:
        raise ValueError(f"config.py 缺少可静态读取的配置项: {missing}")
    return values


def load_expectations(config_path: Path | None = None) -> Expectations:
    """读取项目现有配置，避免复制模型、消融和类别名称。"""
    path = config_path or Path(__file__).with_name("config.py")
    values = _literal_assignments(path)
    tta_sweep = tuple(int(v) for v in values["TTA_SWEEP"])
    if not tta_sweep or max(tta_sweep) <= 0:
        raise ValueError("TTA_SWEEP 必须包含正整数")
    return Expectations(
        classes=tuple(values["CLASS_FOLDERS"].keys()),
        backbones=tuple(values["BACKBONES"].keys()),
        baseline=tuple(values["BASELINE_BACKBONE"].keys()),
        ablations=tuple(values["ABLATIONS"].keys()),
        tta_count=max(tta_sweep),
    )


def core_steps(exp: Expectations) -> tuple[str, ...]:
    """返回 orchestrator 主线中写入 ``ALL.done`` 前的 45 个步骤。"""
    steps = ["manifest", "split", "cache", "image_quality"]
    steps.extend(f"train_{model}" for model in exp.models)
    steps.extend(
        f"pred_{model}_{subset}"
        for model in exp.models
        for subset in ("metacal", "test")
    )
    for ablation in exp.ablations:
        steps.append(f"abl_{ablation}")
        steps.extend(f"pred_abl_{ablation}_{subset}" for subset in ("metacal", "test"))
    steps.extend(
        (
            "frozen_baseline",
            "fusion_analysis",
            "ablation_analysis",
            "gradcam",
            "profile_io",
        )
    )
    return tuple(steps)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_strict_delivery(results_dir: Path, big_dir: Path) -> dict[str, Any]:
    """验收 strict OOF composite results view，不套用旧 45 步语义。"""
    from strict_oof.workflow import EXPECTED_TOTAL_STEPS, EXPECTED_TRAINING_RUNS

    errors: list[dict[str, str]] = []
    checks: list[dict[str, Any]] = []

    def fail(code: str, path: Path, message: str) -> None:
        errors.append({"code": code, "path": str(path), "message": message})

    def load(name: str, expected_type: type) -> Any | None:
        path = results_dir / name
        try:
            value = json.loads(
                path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            fail("strict_json", path, repr(exc))
            return None
        if not isinstance(value, expected_type):
            fail("strict_json_type", path, f"顶层应为 {expected_type.__name__}")
            return None
        return value

    metadata = load("STRICT_DELIVERY.json", dict)
    payload = load("payload_manifest.json", dict)
    contract_hash = metadata.get("contract_sha256") if metadata else None
    if (
        not isinstance(contract_hash, str)
        or len(contract_hash) != 64
        or metadata.get("protocol_id") != "strict_oof_v1"
        or metadata.get("legacy_results_overwritten") is not False
    ):
        fail("strict_metadata", results_dir / "STRICT_DELIVERY.json", "严格协议元数据无效")
    payload_files = payload.get("files", {}) if payload else {}
    if not isinstance(payload_files, dict) or not payload_files:
        fail("strict_manifest", results_dir / "payload_manifest.json", "payload 清单为空")
    else:
        for relative, expected in payload_files.items():
            target = results_dir / relative
            try:
                target.resolve().relative_to(results_dir.resolve())
            except ValueError:
                fail("strict_manifest_path", target, "清单路径越界")
                continue
            if not target.is_file() or _sha256_file(target) != expected:
                fail("strict_manifest_hash", target, "文件缺失或 SHA-256 不匹配")
    strict_report = load("strict_validation_report.json", dict)
    if (
        not strict_report
        or strict_report.get("ok") is not True
        or strict_report.get("contract_sha256") != contract_hash
        or strict_report.get("errors")
        or _sha256_file(results_dir / "strict_validation_report.json")
        != (metadata or {}).get("strict_validation_report_sha256")
    ):
        fail("strict_deep_report", results_dir / "strict_validation_report.json", "严格深验收报告无效")

    split_stats = load("split_stats.json", dict)
    if (
        not split_stats
        or split_stats.get("persons_total") != 8_250
        or split_stats.get("patients")
        != {"test": 1_649, "metacal": 661, "train": 5_940}
    ):
        fail("strict_patient_gate", results_dir / "split_stats.json", "8250=5940+661+1649 硬门失败")
    quality = load("image_quality.json", dict)
    expected_class_sample = 30 * 11
    if (
        not quality
        or quality.get("schema_version") != 2
        or quality.get("protocol", {}).get("per_class_requested") != 30
        or quality.get("sampling", {}).get("attempted") != expected_class_sample
        or quality.get("sampling", {}).get("succeeded") != expected_class_sample
        or quality.get("sampling", {}).get("failed") != 0
        or quality.get("cache_byte_audit", {}).get("all_match") is not True
        or quality.get("cache_byte_audit", {}).get("matches")
        != expected_class_sample
    ):
        fail("strict_image_quality_v2", results_dir / "image_quality.json", "正式 image-quality v2 门失败")

    analysis = load("analysis.json", dict)
    if analysis:
        fusion = analysis.get("fusion_selection_protocol", {})
        nonlinear = analysis.get("nonlinear_stacker_sensitivity", {})
        if (
            analysis.get("status") != "complete"
            or analysis.get("contract_sha256") != contract_hash
            or analysis.get("protocol", {}).get("test_labels_used_for_any_selection")
            is not False
            or fusion.get("selection_partition") != "independent Meta-Cal"
            or set(fusion.get("post_hoc_terminal_test_sensitivity", []))
            != {"stack_mlp_tta1", "stack_gbdt_tta1"}
            or nonlinear.get("used_for_fusion_selection") is not False
            or nonlinear.get("test_labels_used_for_fitting_or_selection") is not False
        ):
            fail("strict_analysis_protocol", results_dir / "analysis.json", "严格 OOF/独立 Meta-Cal 协议字段不完整")
        required = {
            "macro_specificity",
            "weighted_f1",
        }
        final = analysis.get("ensembles", {}).get(analysis.get("final_key"), {})
        if not required.issubset(final):
            fail("strict_analysis_metrics", results_dir / "analysis.json", "主要指标不完整")

    ablation = load("ablation_analysis.json", dict)
    expected_arms = {
        "ce_plain",
        "ce_sampler",
        "focal_only",
        "focal_sampler",
        "full",
        "full_cbreweight",
        "cb_only",
        "mixup_only",
        "cutmix_only",
    }
    if not ablation or {key for key in ablation if not key.startswith("_")} != expected_arms:
        fail("strict_ablation_arms", results_dir / "ablation_analysis.json", "9 臂消融不完整")
    elif any(
        not {
            "flags",
            "metrics",
            "ece",
            "nll",
            "brier_multiclass",
            "minority_overcompensation_index",
            "pred_vs_true_share",
            "evaluation_tta",
        }.issubset(ablation[arm])
        for arm in expected_arms
    ):
        fail("strict_ablation_schema", results_dir / "ablation_analysis.json", "9 臂 schema 不统一")
    elif any(ablation[arm].get("evaluation_tta") != 1 for arm in expected_arms):
        fail("strict_ablation_tta", results_dir / "ablation_analysis.json", "9 臂未统一为 TTA1")

    gradcam = load("gradcam_index.json", list)
    if not gradcam or len(gradcam) < 11:
        fail("strict_gradcam", results_dir / "gradcam_index.json", "Grad-CAM 案例不足")
    else:
        for row in gradcam:
            forbidden = {
                key
                for key in row
                if any(word in str(key).lower() for word in ("patient", "path", "source"))
            }
            target = results_dir / "gradcam" / f"{row.get('case_id')}.png"
            if (
                forbidden
                or not isinstance(row.get("case_id"), str)
                or row.get("tta_component") != 0
                or row.get("annotation_strip_masked_px") != 28
                or not target.is_file()
                or target.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n"
                or _sha256_file(target) != row.get("png_sha256")
            ):
                fail("strict_gradcam_case", target, "Grad-CAM 隐私/协议/PNG 门失败")

    all_done = results_dir / "markers" / "ALL.done"
    core_done = results_dir / "markers" / "core_validation.done"
    if all_done.exists() and not core_done.is_file():
        fail("strict_core_marker", core_done, "ALL.done 存在但 core_validation.done 缺失")
    checks.append(
        {
            "name": "strict_composite_delivery",
            "ok": not errors,
            "detail": {
                "protocol_id": "strict_oof_v1",
                "payload_files": len(payload_files),
                "ablation_arms": len(expected_arms),
                "gradcam_cases": len(gradcam or []),
            },
        }
    )
    return {
        "ok": not errors,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir.resolve()),
        "big_dir": str(big_dir.resolve()),
        "validation_mode": "strict_oof_composite_delivery",
        "contract_sha256": contract_hash,
        "expected_core_steps": EXPECTED_TOTAL_STEPS,
        "expected_training_runs": EXPECTED_TRAINING_RUNS,
        "checks": checks,
        "error_count": len(errors),
        "errors": errors,
    }


class Validator:
    """累积全部失败，避免只报第一个问题。"""

    def __init__(self, results_dir: Path, big_dir: Path, exp: Expectations):
        self.results_dir = results_dir
        self.big_dir = big_dir
        self.exp = exp
        self.errors: list[dict[str, str]] = []
        self.checks: list[dict[str, Any]] = []
        self._prediction_reference: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._subset_images: dict[str, int] = {}

    def cache_path(self, row: dict[str, Any]) -> Path:
        """按 ``build_cache.py`` 的规则推导 manifest 行对应的 JPEG。"""
        source_name = str(row["path"]).replace("\\", "/").rsplit("/", 1)[-1]
        stem = Path(source_name).stem
        return (
            self.big_dir
            / "jpeg384"
            / str(row["cls"])
            / str(row["pt"])
            / (stem + ".jpg")
        )

    @staticmethod
    def source_is_reproducibly_unreadable(path: Path) -> bool:
        """仅把当前仍存在且重复解码失败的源文件视为可审计 QC 排除。"""
        if not path.is_file():
            return False
        for _ in range(2):
            try:
                with Image.open(path) as image:
                    image.load()
            except (OSError, SyntaxError, ValueError):
                continue
            return False
        return True

    def fail(self, code: str, path: Path | str, message: str) -> None:
        self.errors.append({"code": code, "path": str(path), "message": message})

    def record(self, name: str, ok: bool, detail: Any = None) -> None:
        row: dict[str, Any] = {"name": name, "ok": ok}
        if detail is not None:
            row["detail"] = detail
        self.checks.append(row)

    def require_file(self, path: Path, code: str) -> bool:
        if not path.is_file():
            self.fail(code, path, "文件不存在")
            return False
        try:
            size = path.stat().st_size
        except OSError as exc:
            self.fail(code, path, f"无法读取文件状态: {exc}")
            return False
        if size <= 0:
            self.fail(code, path, "文件为空")
            return False
        return True

    def load_json(
        self,
        name: str,
        expected_type: type | tuple[type, ...],
    ) -> Any | None:
        path = self.results_dir / name
        if not self.require_file(path, "json_missing"):
            return None
        try:
            value = json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            self.fail("json_invalid", path, repr(exc))
            return None
        if not isinstance(value, expected_type):
            self.fail(
                "json_type",
                path,
                f"顶层类型应为 {expected_type}，实际为 {type(value).__name__}",
            )
            return None
        if not value:
            self.fail("json_empty", path, "JSON 顶层内容为空")
            return None
        return value

    def require_keys(self, value: dict[str, Any], keys: set[str], path: Path) -> bool:
        missing = sorted(keys - value.keys())
        if missing:
            self.fail("json_keys", path, f"缺少关键字段: {missing}")
            return False
        return True

    def check_markers(self) -> None:
        markers = self.results_dir / "markers"
        steps = core_steps(self.exp)
        if len(steps) != 45:
            self.fail(
                "step_count",
                markers,
                f"按当前配置推导出 {len(steps)} 个核心步骤，预期为 45",
            )
        missing = []
        for name in steps:
            if not self.require_file(markers / f"{name}.done", "marker_missing"):
                missing.append(name)
        core_names = set(steps)
        all_failed = sorted(path.name for path in markers.glob("*.FAILED"))
        failed = [name for name in all_failed if name[: -len(".FAILED")] in core_names]
        optional_failed = [name for name in all_failed if name not in failed]
        if failed:
            self.fail("failed_markers", markers, f"存在 FAILED 标记: {failed}")
        all_done = markers / "ALL.done"
        if not self.require_file(all_done, "all_done_missing"):
            pass
        self.record(
            "markers",
            not missing and not failed and all_done.is_file(),
            {
                "expected": len(steps),
                "missing": missing,
                "failed": failed,
                "non_core_failed_ignored": optional_failed,
            },
        )

    def check_preparation_json(self) -> None:
        manifest = self.load_json("manifest.json", list)
        manifest_patients: set[str] = set()
        if manifest is not None:
            classes = set()
            for index, row in enumerate(manifest):
                if not isinstance(row, dict) or not {"cls", "pt", "path"} <= row.keys():
                    self.fail(
                        "manifest_row",
                        self.results_dir / "manifest.json",
                        f"第 {index} 行缺少 cls/pt/path",
                    )
                    break
                classes.add(row["cls"])
                if isinstance(row["pt"], str) and row["pt"]:
                    manifest_patients.add(row["pt"])
            if classes != set(self.exp.classes):
                self.fail(
                    "manifest_classes",
                    self.results_dir / "manifest.json",
                    f"类别集合不匹配: {sorted(classes)}",
                )

        dataset = self.load_json("dataset_stats.json", dict)
        if dataset is not None:
            path = self.results_dir / "dataset_stats.json"
            self.require_keys(
                dataset,
                {"total_patients", "total_biopsies", "total_images", "per_class"},
                path,
            )
            for key in ("total_patients", "total_biopsies", "total_images"):
                value = dataset.get(key)
                if not isinstance(value, int) or value <= 0:
                    self.fail("dataset_counts", path, f"{key} 必须为正整数")
            per_class = dataset.get("per_class")
            if not isinstance(per_class, dict) or set(per_class) != set(
                self.exp.classes
            ):
                self.fail("dataset_classes", path, "per_class 必须覆盖全部预期类别")
            if manifest is not None:
                if dataset.get("total_images") != len(manifest):
                    self.fail(
                        "dataset_manifest_count",
                        path,
                        "total_images 与 manifest 行数不一致",
                    )
                if dataset.get("total_biopsies") != len(manifest_patients):
                    self.fail(
                        "dataset_manifest_count",
                        path,
                        "total_biopsies 与 manifest 唯一 pt 数不一致",
                    )

        split = self.load_json("split.json", dict)
        if split is not None and set(split.values()) != {"train", "metacal", "test"}:
            self.fail(
                "split_subsets",
                self.results_dir / "split.json",
                "split 必须同时包含 train/metacal/test",
            )
        if (
            split is not None
            and manifest is not None
            and set(split) != manifest_patients
        ):
            self.fail(
                "split_manifest_alignment",
                self.results_dir / "split.json",
                "split 键集合与 manifest 唯一 pt 集合不一致",
            )

        split_stats = self.load_json("split_stats.json", dict)
        if split_stats is not None:
            path = self.results_dir / "split_stats.json"
            self.require_keys(
                split_stats,
                {"patients", "biopsies", "images", "per_class"},
                path,
            )
            for key in ("patients", "biopsies", "images"):
                counts = split_stats.get(key)
                if not isinstance(counts, dict) or set(counts) != {
                    "train",
                    "metacal",
                    "test",
                }:
                    self.fail("split_stats", path, f"{key} 的子集字段不完整")
            image_counts = split_stats.get("images")
            if isinstance(image_counts, dict) and all(
                isinstance(image_counts.get(name), int) and image_counts[name] > 0
                for name in ("train", "metacal", "test")
            ):
                self._subset_images = {
                    name: int(image_counts[name])
                    for name in ("train", "metacal", "test")
                }
                if manifest is not None and split is not None:
                    actual_images = {name: 0 for name in self._subset_images}
                    for row in manifest:
                        if isinstance(row, dict) and row.get("pt") in split:
                            actual_images[split[row["pt"]]] += 1
                    if actual_images != self._subset_images:
                        self.fail(
                            "split_stats_alignment",
                            path,
                            "split_stats.images 与 manifest/split 实际计数不一致",
                        )
            per_class = split_stats.get("per_class")
            if not isinstance(per_class, dict) or set(per_class) != set(
                self.exp.classes
            ):
                self.fail("split_stats_classes", path, "per_class 类别不完整")

        cache = self.load_json("cache_report.json", dict)
        if cache is not None:
            path = self.results_dir / "cache_report.json"
            self.require_keys(
                cache,
                {
                    "total",
                    "converted",
                    "skipped",
                    "errors",
                    "img_size",
                    "jpeg_quality",
                    "annotation_strip_masked_px",
                },
                path,
            )
            if cache.get("img_size") != 384:
                self.fail("cache_img_size", path, "img_size 必须为 384")
            if cache.get("jpeg_quality") != 92:
                self.fail("cache_jpeg_quality", path, "jpeg_quality 必须为 92")
            if cache.get("annotation_strip_masked_px") != 28:
                self.fail(
                    "cache_mask_rows",
                    path,
                    "annotation_strip_masked_px 必须为 28",
                )
            converted = cache.get("converted")
            skipped = cache.get("skipped")
            errors = cache.get("errors")
            total = cache.get("total")
            if not isinstance(converted, int) or not isinstance(skipped, int):
                successful = None
                self.fail("cache_counts", path, "converted/skipped 必须为整数")
            else:
                successful = converted + skipped
            if successful is None or successful <= 0:
                self.fail("cache_counts", path, "converted + skipped 必须大于 0")
            if (
                not isinstance(total, int)
                or not isinstance(errors, list)
                or successful is None
                or total != successful + len(errors)
            ):
                self.fail("cache_counts", path, "total 与成功数及 errors 数量不闭合")
            exclusions_valid = isinstance(errors, list)
            excluded_rows: list[dict[str, Any]] = []
            if isinstance(errors, list) and errors:
                if len(errors) > MAX_CACHE_QC_EXCLUSIONS:
                    exclusions_valid = False
                    self.fail(
                        "cache_errors",
                        path,
                        f"QC 排除 {len(errors)} 张，超过项目上限 "
                        f"{MAX_CACHE_QC_EXCLUSIONS}",
                    )
                manifest_by_source = {
                    str(row.get("path")): row
                    for row in manifest or []
                    if isinstance(row, dict) and isinstance(row.get("path"), str)
                }
                seen_sources: set[str] = set()
                for index, error in enumerate(errors):
                    if not isinstance(error, dict):
                        exclusions_valid = False
                        self.fail("cache_errors", path, f"errors[{index}] 不是对象")
                        continue
                    source = error.get("src")
                    detail = error.get("err")
                    row = (
                        manifest_by_source.get(source)
                        if isinstance(source, str)
                        else None
                    )
                    if (
                        row is None
                        or not isinstance(detail, str)
                        or not detail.startswith("ERR:")
                        or source in seen_sources
                    ):
                        exclusions_valid = False
                        self.fail(
                            "cache_errors",
                            path,
                            f"errors[{index}] 不是唯一且可追溯的 manifest 转换失败",
                        )
                        continue
                    seen_sources.add(source)
                    cache_target = self.cache_path(row)
                    if cache_target.exists():
                        exclusions_valid = False
                        self.fail(
                            "cache_errors",
                            path,
                            f"errors[{index}] 对应 JPEG 实际仍存在",
                        )
                        continue
                    if not self.source_is_reproducibly_unreadable(Path(source)):
                        exclusions_valid = False
                        self.fail(
                            "cache_errors",
                            path,
                            f"errors[{index}] 的源文件不存在或当前可以正常解码",
                        )
                        continue
                    excluded_rows.append(row)
            if (
                manifest is not None
                and isinstance(errors, list)
                and successful != len(manifest) - len(errors)
            ):
                self.fail(
                    "cache_manifest_count",
                    path,
                    "缓存成功数与 manifest 减去 QC 排除后的行数不一致",
                )
            cache_dir = self.big_dir / "jpeg384"
            if not cache_dir.is_dir():
                self.fail("cache_dir", cache_dir, "JPEG 缓存目录不存在")
            elif successful is not None:
                actual = 0
                try:
                    actual = sum(
                        1
                        for item in cache_dir.rglob("*.jpg")
                        if item.is_file() and item.stat().st_size > 0
                    )
                except OSError as exc:
                    self.fail("cache_read", cache_dir, repr(exc))
                if actual != successful:
                    self.fail(
                        "cache_count",
                        cache_dir,
                        f"非空 JPEG 数 {actual} != 报告成功数 {successful}",
                    )
            if manifest is not None:
                targets = [self.cache_path(row) for row in manifest]
                if len(set(targets)) != len(targets):
                    self.fail(
                        "cache_path_collision",
                        path,
                        "不同 manifest 行映射到了同一 JPEG 路径",
                    )
                excluded_sources = {str(row["path"]) for row in excluded_rows}
                missing_unreported = [
                    target
                    for row, target in zip(manifest, targets, strict=True)
                    if str(row["path"]) not in excluded_sources
                    and (not target.is_file() or target.stat().st_size <= 0)
                ]
                if missing_unreported:
                    self.fail(
                        "cache_missing",
                        cache_dir,
                        f"有 {len(missing_unreported)} 个未申报 QC 排除的 JPEG 缺失或为空",
                    )
            if errors and exclusions_valid and len(excluded_rows) == len(errors):
                excluded_by_subset = {name: 0 for name in self._subset_images}
                if split is None:
                    exclusions_valid = False
                    self.fail("cache_errors", path, "无法按 split 核对 QC 排除")
                else:
                    for row in excluded_rows:
                        subset = split.get(row["pt"])
                        if subset not in excluded_by_subset:
                            exclusions_valid = False
                            self.fail(
                                "cache_errors", path, "QC 排除无法映射到有效 split"
                            )
                            break
                        excluded_by_subset[subset] += 1
                if exclusions_valid:
                    self._subset_images = {
                        name: count - excluded_by_subset[name]
                        for name, count in self._subset_images.items()
                    }
                    self.record(
                        "cache_qc_exclusions",
                        True,
                        {
                            "count": len(excluded_rows),
                            "by_subset": excluded_by_subset,
                        },
                    )

        quality = self.load_json("image_quality.json", dict)
        if quality is not None:
            path = self.results_dir / "image_quality.json"
            self.require_keys(
                quality,
                {
                    "schema_version",
                    "protocol",
                    "sampling",
                    "cache_byte_audit",
                    "mask_audit",
                    "overall",
                    "per_class",
                },
                path,
            )
            if quality.get("schema_version") != 2:
                self.fail(
                    "quality_schema",
                    path,
                    "须为 resize→mask→JPEG 真实输入链的 v2 报告",
                )
            protocol = quality.get("protocol")
            if not isinstance(protocol, dict) or protocol.get(
                "per_class_requested"
            ) != 30:
                self.fail(
                    "quality_protocol",
                    path,
                    "per_class_requested 必须为 30",
                )
            sampling = quality.get("sampling")
            if not isinstance(sampling, dict):
                self.fail("quality_sampling", path, "sampling 不是对象")
                sample_n = None
            else:
                sample_n = sampling.get("succeeded")
                if not isinstance(sample_n, int) or sample_n <= 0:
                    self.fail("quality_sampling", path, "succeeded 必须为正整数")
                if sampling.get("failed") != 0:
                    self.fail(
                        "quality_sampling_errors",
                        path,
                        f"抽样处理错误数为 {sampling.get('failed')}",
                    )
                expected_n = 30 * len(self.exp.classes)
                if sampling.get("attempted") != expected_n or sample_n != expected_n:
                    self.fail(
                        "quality_sampling_count",
                        path,
                        f"必须完整分析 {expected_n} 张（30/类）",
                    )
                expected_per_class = {name: 30 for name in self.exp.classes}
                if sampling.get("selected_by_class") != expected_per_class:
                    self.fail(
                        "quality_selected_by_class",
                        path,
                        "selected_by_class 不是每类 30 张",
                    )
                if sampling.get("successful_by_class") != expected_per_class:
                    self.fail(
                        "quality_success_by_class",
                        path,
                        "successful_by_class 不是每类 30 张",
                    )
            cache_audit = quality.get("cache_byte_audit")
            if not isinstance(cache_audit, dict) or not cache_audit.get(
                "all_match"
            ):
                self.fail(
                    "quality_cache_bytes",
                    path,
                    "重建 JPEG 与训练实际缓存字节未全部一致",
                )
            elif (
                cache_audit.get("matches") != 30 * len(self.exp.classes)
                or cache_audit.get("mismatches") != 0
            ):
                self.fail(
                    "quality_cache_byte_counts",
                    path,
                    "JPEG 字节审计数未覆盖全部 30/类样本",
                )
            mask_audit = quality.get("mask_audit")
            mask_fraction = (
                mask_audit.get("configured_area_fraction")
                if isinstance(mask_audit, dict)
                else None
            )
            if not isinstance(mask_fraction, (int, float)) or not math.isclose(
                float(mask_fraction), 28 / 384, rel_tol=0, abs_tol=1e-12
            ):
                self.fail(
                    "quality_mask_fraction",
                    path,
                    "遮蔽面积比不是 28/384",
                )
            changed_summary = (
                mask_audit.get("changed_pixel_fraction")
                if isinstance(mask_audit, dict)
                else None
            )
            if (
                not isinstance(changed_summary, dict)
                or changed_summary.get("n") != 30 * len(self.exp.classes)
                or changed_summary.get("n_nonfinite") != 0
                or not isinstance(changed_summary.get("mean"), (int, float))
                or not 0 <= float(changed_summary["mean"]) <= 28 / 384
            ):
                self.fail(
                    "quality_mask_changed_fraction",
                    path,
                    "遮蔽实际改变像素比例摘要无效",
                )
            per_class = quality.get("per_class")
            if not isinstance(per_class, dict) or set(per_class) != set(
                self.exp.classes
            ):
                self.fail("quality_classes", path, "per_class 类别不完整")
            expected_stages = {
                "resize_only_vs_native",
                "resize_mask_vs_native",
                "network_input_vs_native",
                "jpeg_vs_resize_mask",
            }
            overall = quality.get("overall")
            if not isinstance(overall, dict) or set(overall) != expected_stages:
                self.fail("quality_stages", path, "overall 未覆盖四个预处理比较")
            else:
                for stage, metrics in overall.items():
                    if not isinstance(metrics, dict) or set(metrics) != {
                        "ssim",
                        "psnr_db",
                        "lpips",
                    }:
                        self.fail(
                            "quality_metrics",
                            path,
                            f"{stage} 未同时报告 SSIM/PSNR/LPIPS",
                        )
                        continue
                    for metric, summary in metrics.items():
                        if (
                            not isinstance(summary, dict)
                            or summary.get("n") != sample_n
                            or summary.get("n_nonfinite") != 0
                            or not isinstance(summary.get("mean"), (int, float))
                            or not isinstance(summary.get("sd"), (int, float))
                        ):
                            self.fail(
                                "quality_metric_summary",
                                path,
                                f"{stage}/{metric} 摘要无效或样本数不一致",
                            )
            if isinstance(per_class, dict) and set(per_class) == set(
                self.exp.classes
            ):
                for class_name, stages in per_class.items():
                    if not isinstance(stages, dict) or set(stages) != expected_stages:
                        self.fail(
                            "quality_class_stages",
                            path,
                            f"{class_name} 未覆盖四个预处理比较",
                        )
                        continue
                    for stage, metrics in stages.items():
                        if not isinstance(metrics, dict) or set(metrics) != {
                            "ssim",
                            "psnr_db",
                            "lpips",
                        }:
                            self.fail(
                                "quality_class_metrics",
                                path,
                                f"{class_name}/{stage} 指标不完整",
                            )
                            continue
                        for metric, summary in metrics.items():
                            if (
                                not isinstance(summary, dict)
                                or summary.get("n") != 30
                                or summary.get("n_nonfinite") != 0
                                or not isinstance(summary.get("mean"), (int, float))
                                or not isinstance(summary.get("sd"), (int, float))
                            ):
                                self.fail(
                                    "quality_class_summary",
                                    path,
                                    f"{class_name}/{stage}/{metric} 摘要无效",
                                )

    def check_ablation_backbone(self) -> str | None:
        data = self.load_json("ablation_backbone.json", dict)
        if data is None:
            return None
        path = self.results_dir / "ablation_backbone.json"
        if not self.require_keys(data, {"backbone", "metacal_macro_f1"}, path):
            return None
        backbone = data.get("backbone")
        scores = data.get("metacal_macro_f1")
        if backbone not in self.exp.backbones:
            self.fail("ablation_backbone", path, f"非法骨干: {backbone!r}")
            return None
        if not isinstance(scores, dict) or set(scores) != set(self.exp.backbones):
            self.fail("ablation_scores", path, "Meta-Cal 分数未覆盖五个主骨干")
        return str(backbone)

    def check_training(self, ablation_backbone: str | None) -> None:
        tags = list(self.exp.models)
        if ablation_backbone is not None:
            tags.extend(
                f"{ablation_backbone}_abl_{name}" for name in self.exp.ablations
            )
        else:
            tags.extend(f"UNKNOWN_abl_{name}" for name in self.exp.ablations)
        for tag in tags:
            checkpoint = self.big_dir / "checkpoints" / f"{tag}.pt"
            if self.require_file(checkpoint, "checkpoint_missing"):
                try:
                    if not zipfile.is_zipfile(checkpoint):
                        raise zipfile.BadZipFile("不是 PyTorch ZIP 序列化文件")
                    with zipfile.ZipFile(checkpoint) as archive:
                        names = archive.namelist()
                        bad_member = archive.testzip()
                        if bad_member:
                            raise zipfile.BadZipFile(f"成员 CRC 校验失败: {bad_member}")
                        if not any(name.endswith("/data.pkl") for name in names):
                            raise zipfile.BadZipFile("缺少 data.pkl")
                        if not any(name.endswith("/version") for name in names):
                            raise zipfile.BadZipFile("缺少 version")
                except (OSError, zipfile.BadZipFile) as exc:
                    self.fail("checkpoint_invalid", checkpoint, repr(exc))
            data = self.load_json(f"train_log_{tag}.json", dict)
            if data is None:
                continue
            path = self.results_dir / f"train_log_{tag}.json"
            keys = {
                "tag",
                "model_name",
                "best_epoch",
                "best_metacal_macro_f1",
                "class_counts_train",
                "epochs",
                "train_images",
                "metacal_images",
            }
            self.require_keys(data, keys, path)
            if data.get("tag") != tag:
                self.fail("train_tag", path, f"tag 应为 {tag!r}")
            counts = data.get("class_counts_train")
            if not isinstance(counts, list) or len(counts) != len(self.exp.classes):
                self.fail("train_classes", path, "class_counts_train 维度错误")
            elif not all(
                isinstance(count, int) and count >= 0 for count in counts
            ) or sum(counts) != data.get("train_images"):
                self.fail(
                    "train_classes",
                    path,
                    "class_counts_train 必须为非负整数且总和等于 train_images",
                )
            epochs = data.get("epochs")
            if not isinstance(epochs, list) or not epochs:
                self.fail("train_epochs", path, "epochs 为空")
            score = data.get("best_metacal_macro_f1")
            if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                self.fail("train_score", path, "best_metacal_macro_f1 非法")
            for key, subset in (
                ("train_images", "train"),
                ("metacal_images", "metacal"),
            ):
                if (
                    subset in self._subset_images
                    and data.get(key) != self._subset_images[subset]
                ):
                    self.fail(
                        "train_split_count",
                        path,
                        f"{key} 与 split_stats.images.{subset} 不一致",
                    )

    def check_prediction(self, tag: str, subset: str) -> None:
        path = self.big_dir / "preds" / f"{tag}__{subset}.npz"
        if not self.require_file(path, "prediction_missing"):
            return
        try:
            with np.load(path, allow_pickle=False) as data:
                required = {
                    "probs",
                    "labels",
                    "patients",
                    "ms_per_image_single",
                    "n_images",
                    "secs_per_pass",
                }
                missing = sorted(required - set(data.files))
                if missing:
                    self.fail("prediction_keys", path, f"缺少数组: {missing}")
                    return
                probs = np.asarray(data["probs"])
                labels = np.asarray(data["labels"])
                patients = np.asarray(data["patients"])
                timing = float(np.asarray(data["ms_per_image_single"]).reshape(()))
                n_recorded = int(np.asarray(data["n_images"]).reshape(()))
                secs_per_pass = float(np.asarray(data["secs_per_pass"]).reshape(()))
        except Exception as exc:  # noqa: BLE001  损坏 NPZ 的异常类型不固定
            self.fail("prediction_invalid", path, repr(exc))
            return

        expected_shape = (self.exp.tta_count, None, len(self.exp.classes))
        if (
            probs.ndim != 3
            or probs.shape[0] != expected_shape[0]
            or probs.shape[2] != expected_shape[2]
            or probs.shape[1] <= 0
        ):
            self.fail(
                "prediction_shape",
                path,
                f"probs 形状 {probs.shape}，预期 ({self.exp.tta_count}, N, "
                f"{len(self.exp.classes)}) 且 N>0",
            )
            return
        n_images = probs.shape[1]
        if labels.shape != (n_images,) or patients.shape != (n_images,):
            self.fail(
                "prediction_alignment",
                path,
                f"labels={labels.shape}, patients={patients.shape}, N={n_images}",
            )
            return
        if not np.issubdtype(labels.dtype, np.integer):
            self.fail("prediction_labels", path, "labels 必须为整数数组")
        elif labels.min() < 0 or labels.max() >= len(self.exp.classes):
            self.fail("prediction_labels", path, "labels 超出类别范围")
        elif set(labels.tolist()) != set(range(len(self.exp.classes))):
            self.fail("prediction_labels", path, "预测子集未覆盖全部预期类别")
        if patients.dtype.kind not in {"U", "S"} or np.any(patients == ""):
            self.fail("prediction_patients", path, "patients 必须为非空字符串数组")
        if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
            self.fail("prediction_probs", path, "概率包含 NaN/Inf 或超出 [0,1]")
        else:
            sums = probs.astype(np.float32).sum(axis=2)
            if not np.allclose(sums, 1.0, atol=0.02, rtol=0.0):
                self.fail("prediction_probs", path, "类别概率和未近似等于 1")
        if not math.isfinite(timing) or timing <= 0:
            self.fail("prediction_timing", path, "ms_per_image_single 必须为正数")
        if n_recorded != n_images:
            self.fail("prediction_count", path, "n_images 与 probs 的 N 不一致")
        if subset in self._subset_images and n_images != self._subset_images[subset]:
            self.fail(
                "prediction_split_count",
                path,
                f"预测 N 与 split_stats.images.{subset} 不一致",
            )
        if not math.isfinite(secs_per_pass) or secs_per_pass <= 0:
            self.fail("prediction_timing", path, "secs_per_pass 必须为正数")

        reference = self._prediction_reference.get(subset)
        if reference is None:
            self._prediction_reference[subset] = (labels.copy(), patients.copy())
        elif not (
            np.array_equal(labels, reference[0])
            and np.array_equal(patients, reference[1])
        ):
            self.fail(
                "prediction_order",
                path,
                f"{subset} 的标签或患者顺序与其他模型不一致",
            )

        meta = self.load_json(f"predict_meta_{tag}__{subset}.json", dict)
        if meta is not None:
            meta_path = self.results_dir / f"predict_meta_{tag}__{subset}.json"
            self.require_keys(
                meta, {"tag", "subset", "n", "ms_per_image_single_pass"}, meta_path
            )
            if (
                meta.get("tag") != tag
                or meta.get("subset") != subset
                or meta.get("n") != n_images
            ):
                self.fail("prediction_meta", meta_path, "tag/subset/n 与 NPZ 不一致")

    def check_predictions(self, ablation_backbone: str | None) -> None:
        tags = list(self.exp.models)
        if ablation_backbone is not None:
            tags.extend(
                f"{ablation_backbone}_abl_{name}" for name in self.exp.ablations
            )
        for tag in tags:
            for subset in ("metacal", "test"):
                self.check_prediction(tag, subset)

    def check_frozen(self) -> None:
        data = self.load_json("frozen_baseline.json", dict)
        if data is None:
            return
        path = self.results_dir / "frozen_baseline.json"
        self.require_keys(data, {"feature_dim", "logreg", "random_forest"}, path)
        for model in ("logreg", "random_forest"):
            row = data.get(model)
            if (
                not isinstance(row, dict)
                or not {
                    "accuracy",
                    "macro_f1",
                    "auc_ovr_macro",
                }
                <= row.keys()
            ):
                self.fail("frozen_metrics", path, f"{model} 指标不完整")
                continue
            for key in ("accuracy", "macro_f1", "auc_ovr_macro"):
                value = row[key]
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    self.fail("frozen_metrics", path, f"{model}.{key} 不是有限数值")

    def check_analysis(self) -> None:
        data = self.load_json("analysis.json", dict)
        if data is None:
            return
        path = self.results_dir / "analysis.json"
        keys = {
            "singles",
            "voting_weights",
            "tta_sweep",
            "ensembles",
            "final_key",
            "selection_on_metacal",
            "fusion_selection_protocol",
            "best_single_selection",
            "calibration",
            "referral",
            "final_ci",
            "per_class_ci",
            "confusion_matrix",
            "pr_curves",
            "patient_level",
            "dca",
            "inference_cost",
        }
        self.require_keys(data, keys, path)
        singles = data.get("singles")
        if not isinstance(singles, dict) or set(singles) != set(self.exp.models):
            self.fail("analysis_models", path, "singles 未覆盖全部主模型和基线")
        weights = data.get("voting_weights")
        if not isinstance(weights, dict) or set(weights) != set(self.exp.backbones):
            self.fail("analysis_weights", path, "voting_weights 未覆盖五个主骨干")
        ensembles = data.get("ensembles")
        final_key = data.get("final_key")
        if (
            not isinstance(ensembles, dict)
            or not ensembles
            or final_key not in ensembles
        ):
            self.fail("analysis_final", path, "final_key 未指向非空 ensembles 项")
        selection = data.get("selection_on_metacal")
        protocol = data.get("fusion_selection_protocol")
        expected_candidates = {"voting_tta8", "stack_lr_tta8"}
        expected_sensitivity = {"stack_mlp_tta1", "stack_gbdt_tta1"}
        candidate_roles = (
            protocol.get("candidate_roles") if isinstance(protocol, dict) else None
        )
        if (
            not isinstance(protocol, dict)
            or protocol.get("splitter") != "StratifiedGroupKFold"
            or protocol.get("n_splits") != 5
            or protocol.get("test_labels_used_for_selection") is not False
            or protocol.get("selected_candidate") != final_key
            or not isinstance(protocol.get("folds"), list)
            or len(protocol["folds"]) != 5
            or set(protocol.get("oof_selection_candidates", [])) != expected_candidates
            or set(protocol.get("post_hoc_terminal_test_sensitivity", []))
            != expected_sensitivity
            or not isinstance(candidate_roles, dict)
            or any(
                candidate_roles.get(name, {}).get("used_for_fusion_selection")
                is not True
                for name in expected_candidates
            )
            or any(
                candidate_roles.get(name, {}).get("used_for_fusion_selection")
                is not False
                for name in expected_sensitivity
            )
        ):
            self.fail(
                "analysis_fusion_oof",
                path,
                "融合选型不是固定的患者分组 5 折 OOF 协议",
            )
        candidate_scores = (
            protocol.get("candidate_oof_scores") if isinstance(protocol, dict) else None
        )
        if (
            not isinstance(selection, dict)
            or set(selection) != expected_candidates
            or selection != candidate_scores
            or any(
                not isinstance(row, dict)
                or row.get("estimate") != "patient_grouped_oof"
                or not isinstance(row.get("macro_f1"), (int, float))
                for row in selection.values()
            )
        ):
            self.fail(
                "analysis_fusion_oof",
                path,
                "融合候选缺少完整的患者分组 OOF 分数",
            )
        best_single = data.get("best_single_selection")
        if (
            not isinstance(best_single, dict)
            or best_single.get("selected") not in self.exp.backbones
            or best_single.get("test_labels_used_for_selection") is not False
            or not isinstance(best_single.get("metacal_macro_f1"), dict)
            or set(best_single["metacal_macro_f1"]) != set(self.exp.backbones)
        ):
            self.fail(
                "analysis_best_single_selection",
                path,
                "最佳单骨干未由 Meta-Cal 完整预选",
            )
        calibration = data.get("calibration")
        referral = data.get("referral")
        if (
            not isinstance(calibration, dict)
            or calibration.get("fit_source")
            != "selected candidate's patient-grouped meta-cal OOF probabilities"
        ):
            self.fail(
                "analysis_calibration_oof",
                path,
                "温度参数不是基于选中候选的 Meta-Cal OOF 概率拟合",
            )
        if not isinstance(referral, dict) or "patient-grouped OOF" not in referral.get(
            "metacal_probability_source", ""
        ):
            self.fail(
                "analysis_referral_oof",
                path,
                "转诊阈值不是基于 Meta-Cal OOF 概率选择",
            )
        expected_classes = set(self.exp.classes)
        for key in ("per_class_ci", "pr_curves", "dca"):
            value = data.get(key)
            if not isinstance(value, dict) or set(value) != expected_classes:
                self.fail("analysis_classes", path, f"{key} 类别不完整")
        for key in (
            "tta_sweep",
            "calibration",
            "final_ci",
            "patient_level",
            "inference_cost",
        ):
            value = data.get(key)
            if not isinstance(value, dict) or not value:
                self.fail("analysis_empty", path, f"{key} 缺失或为空")
        patient_level = data.get("patient_level")
        if isinstance(patient_level, dict):
            total = patient_level.get("n_patients_test")
            evaluated = patient_level.get("n_patients_evaluated")
            mixed = patient_level.get("n_patients_mixed_diagnoses_excluded")
            if (
                not all(isinstance(value, int) for value in (total, evaluated, mixed))
                or evaluated + mixed != total
            ):
                self.fail(
                    "analysis_patient_count",
                    path,
                    "患者聚合分母不完整或 evaluated + mixed != total",
                )
        cm = data.get("confusion_matrix")
        n = len(self.exp.classes)
        if not isinstance(cm, dict) or cm.get("labels") != list(self.exp.classes):
            self.fail("analysis_confusion", path, "混淆矩阵标签与配置不一致")
        else:
            for key in ("counts", "row_normalized"):
                matrix = cm.get(key)
                if (
                    not isinstance(matrix, list)
                    or len(matrix) != n
                    or any(not isinstance(row, list) or len(row) != n for row in matrix)
                ):
                    self.fail("analysis_confusion", path, f"{key} 不是 {n}x{n}")

    def check_ablation(self, ablation_backbone: str | None) -> None:
        data = self.load_json("ablation_analysis.json", dict)
        if data is None:
            return
        path = self.results_dir / "ablation_analysis.json"
        if set(data) != set(self.exp.ablations) | {"_backbone"}:
            self.fail("ablation_rows", path, "消融结果未精确覆盖六个消融臂")
        if ablation_backbone is not None and data.get("_backbone") != ablation_backbone:
            self.fail("ablation_backbone_mismatch", path, "消融骨干与选择结果不一致")
        for name in self.exp.ablations:
            row = data.get(name)
            if not isinstance(row, dict):
                continue
            self.require_keys(
                row,
                {"flags", "metrics", "ece", "nll", "pred_vs_true_share"},
                path,
            )
            flags = row.get("flags")
            if not isinstance(flags, dict) or set(flags) != {
                "sampler",
                "focal",
                "mix",
                "cb_reweight",
            }:
                self.fail("ablation_flags", path, f"{name}.flags 字段不完整")
            metrics = row.get("metrics")
            shares = row.get("pred_vs_true_share")
            per_class = metrics.get("per_class") if isinstance(metrics, dict) else None
            if not isinstance(per_class, dict) or set(per_class) != set(
                self.exp.classes
            ):
                self.fail("ablation_classes", path, f"{name}.metrics.per_class 不完整")
            if not isinstance(shares, dict) or set(shares) != set(self.exp.classes):
                self.fail("ablation_classes", path, f"{name}.pred_vs_true_share 不完整")

    def check_gradcam(self) -> None:
        data = self.load_json("gradcam_index.json", list)
        if data is None:
            return
        path = self.results_dir / "gradcam_index.json"
        image_dir = self.results_dir / "gradcam"
        correct_classes: set[str] = set()
        missed_classes: set[str] = set()
        cases: set[str] = set()
        for index, row in enumerate(data):
            if (
                not isinstance(row, dict)
                or not {
                    "case",
                    "kind",
                    "true",
                    "pred",
                    "conf",
                    "patient",
                }
                <= row.keys()
            ):
                self.fail("gradcam_row", path, f"第 {index} 行关键字段不完整")
                continue
            case = row["case"]
            if not isinstance(case, str) or not case or case in cases:
                self.fail("gradcam_case", path, f"第 {index} 行 case 为空或重复")
                continue
            cases.add(case)
            if (
                row["true"] not in self.exp.classes
                or row["pred"] not in self.exp.classes
            ):
                self.fail("gradcam_class", path, f"第 {index} 行类别非法")
            if row.get("kind") == "correct" and row["true"] == row["pred"]:
                correct_classes.add(row["true"])
            elif row.get("kind") == "missed_as_IgAN" and row["pred"] == "IgAN":
                missed_classes.add(row["true"])
            else:
                self.fail("gradcam_kind", path, f"第 {index} 行 kind/预测不一致")
            image = image_dir / f"{case}.png"
            if not self.require_file(image, "gradcam_image"):
                continue
            try:
                signature = image.read_bytes()[:8]
            except OSError as exc:
                self.fail("gradcam_image", image, repr(exc))
                continue
            if signature != b"\x89PNG\r\n\x1a\n":
                self.fail("gradcam_image", image, "文件不是有效 PNG 签名")
        # 正文 Fig. 8 固定使用 6 个不同类别的正确案例，并需要少数类误判为
        # IgAN 的失败案例；补充图还需要主图选例之外至少一个额外面板。
        if len(correct_classes) < 6:
            self.fail(
                "gradcam_coverage",
                path,
                f"正确案例仅覆盖 {len(correct_classes)} 类，Fig. 8 至少需要 6 类",
            )
        if len(missed_classes) < 2:
            self.fail(
                "gradcam_coverage",
                path,
                "少数类误判为 IgAN 的案例不足 2 个不同真实类别",
            )
        if len(cases) < 11:
            self.fail(
                "gradcam_coverage",
                path,
                f"Grad-CAM 仅 {len(cases)} 个案例，无法稳健生成主图与补充图",
            )

    def check_profile(self) -> None:
        data = self.load_json("io_profile.json", dict)
        if data is None:
            return
        path = self.results_dir / "io_profile.json"
        self.require_keys(data, {"batch", "steps_per_repeat", "repeats", "modes"}, path)
        modes = data.get("modes")
        if not isinstance(modes, dict):
            self.fail("profile_modes", path, "modes 必须为对象")
            return
        for mode in ("governance_on", "governance_off"):
            row = modes.get(mode)
            if not isinstance(row, dict) or not row:
                self.fail("profile_modes", path, f"{mode} 结果缺失或为空")
                continue
            for key in ("repeats", "mean_step_s", "gpu_util_mean", "cpu_util_mean"):
                if key not in row:
                    self.fail("profile_keys", path, f"{mode} 缺少 {key}")
            if not isinstance(row.get("repeats"), list) or not row.get("repeats"):
                self.fail("profile_repeats", path, f"{mode}.repeats 为空")
            for key in ("mean_step_s", "gpu_util_mean", "cpu_util_mean"):
                value = row.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    self.fail("profile_value", path, f"{mode}.{key} 不是有限数值")
            profiler = self.results_dir / f"profiler_{mode}.txt"
            self.require_file(profiler, "profiler_missing")

    def run(self) -> dict[str, Any]:
        """运行全部只读检查并返回可序列化报告。"""
        self.check_markers()
        self.check_preparation_json()
        ablation_backbone = self.check_ablation_backbone()
        self.check_training(ablation_backbone)
        self.check_predictions(ablation_backbone)
        self.check_frozen()
        self.check_analysis()
        self.check_ablation(ablation_backbone)
        self.check_gradcam()
        self.check_profile()
        return {
            "ok": not self.errors,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "results_dir": str(self.results_dir.resolve()),
            "big_dir": str(self.big_dir.resolve()),
            "expected_core_steps": len(core_steps(self.exp)),
            "expected_classes": list(self.exp.classes),
            "checks": self.checks,
            "error_count": len(self.errors),
            "errors": self.errors,
        }


def validate(results_dir: Path, big_dir: Path) -> dict[str, Any]:
    """公开入口：验证指定目录并返回报告，不写任何实验产物。"""
    if (results_dir / "STRICT_DELIVERY.json").is_file():
        try:
            return validate_strict_delivery(results_dir, big_dir)
        except Exception as exc:  # noqa: BLE001  strict 验收器也必须失败关闭
            try:
                from strict_oof.workflow import (
                    EXPECTED_TOTAL_STEPS as strict_steps,
                    EXPECTED_TRAINING_RUNS as strict_runs,
                )
            except ImportError:
                strict_steps, strict_runs = 94, 39
            return {
                "ok": False,
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "results_dir": str(results_dir.resolve()),
                "big_dir": str(big_dir.resolve()),
                "validation_mode": "strict_oof_composite_delivery",
                "expected_core_steps": strict_steps,
                "expected_training_runs": strict_runs,
                "checks": [],
                "error_count": 1,
                "errors": [
                    {
                        "code": "strict_validator_exception",
                        "path": str(Path(__file__)),
                        "message": repr(exc),
                    }
                ],
            }
    try:
        expectations = load_expectations()
    except (OSError, SyntaxError, ValueError) as exc:
        return {
            "ok": False,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "results_dir": str(results_dir.resolve()),
            "big_dir": str(big_dir.resolve()),
            "expected_core_steps": 45,
            "expected_classes": [],
            "checks": [],
            "error_count": 1,
            "errors": [
                {
                    "code": "config_invalid",
                    "path": str(Path(__file__).with_name("config.py")),
                    "message": repr(exc),
                }
            ],
        }
    validator = Validator(results_dir, big_dir, expectations)
    try:
        return validator.run()
    except Exception as exc:  # noqa: BLE001  验收器自身异常也必须 fail-closed
        validator.fail(
            "validator_exception",
            Path(__file__),
            f"未处理的校验异常: {exc!r}",
        )
        return {
            "ok": False,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "results_dir": str(results_dir.resolve()),
            "big_dir": str(big_dir.resolve()),
            "expected_core_steps": len(core_steps(expectations)),
            "expected_classes": list(expectations.classes),
            "checks": validator.checks,
            "error_count": len(validator.errors),
            "errors": validator.errors,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析 CLI 参数。"""
    here = Path(__file__).resolve().parent
    default_exp_dir = Path(os.environ.get("SR_EXP_DIR", here))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=default_exp_dir / "results")
    parser.add_argument(
        "--big-dir",
        type=Path,
        default=default_exp_dir / "big",
        help="默认与 config.BIG_DIR 相同，即 <SR_EXP_DIR>/big",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="报告路径；默认写入 <results-dir>/core_validation.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口。"""
    args = parse_args(argv)
    json_out = args.json_out or args.results_dir / "core_validation.json"
    report = validate(args.results_dir, args.big_dir)
    try:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        report["ok"] = False
        print(f"FAIL: 无法写入验证报告 {json_out}: {exc}", file=sys.stderr)
        return 1
    status = "PASS" if report["ok"] else "FAIL"
    print(
        f"{status}: 核心结果完整性检查，错误 {report['error_count']} 项；"
        f"报告: {json_out}"
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
