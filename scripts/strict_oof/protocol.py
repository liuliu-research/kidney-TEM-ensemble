"""严格 OOF 协议、路径与不变量。

该命名空间与原始 ``results/`` / ``big/`` 产物完全隔离。只有
``strict_oof/READY.json`` 通过后，上层稿件构建器才可显式切换数据源。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

import config as legacy
from strict_oof import PROTOCOL_ID

OUTER_FOLDS = 5
INNER_FOLDS = 5
INNER_SELECTION_RULE = "outer_fold_mod_inner_folds"
TERMINAL_EPOCH_RULE = "median_of_five_outer_best_epochs"
OOF_TTA_COUNT = 8
FINAL_TTA_COUNT = 16
SCHEDULER_HORIZON_EPOCHS = 40
BACKBONES = tuple(legacy.BACKBONES)
TRAIN_MODELS = BACKBONES + ("resnet50",)
CLASSES = tuple(legacy.CLASSES)
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASSES)}
MODEL_ARCHITECTURES = {
    **{str(key): str(value) for key, value in legacy.BACKBONES.items()},
    **{str(key): str(value) for key, value in legacy.BASELINE_BACKBONE.items()},
}
IMAGE_SIZE = int(legacy.IMG_SIZE)
JPEG_QUALITY = int(legacy.JPEG_QUALITY)
ABLATION_EPOCH_BUDGET = int(legacy.ABLATION_EPOCH_BUDGET)
SEED = int(legacy.SEED)
EXPECTED_RAW_IMAGES = 60_512
EXPECTED_USABLE_IMAGES = 60_509
EXPECTED_QC_PATH_SHA256 = {
    "4db490f0d8bb0fe3a5c90a797abf8652a1a5cd12a05fef97288665d5798c4041",
    "243fead79fbd69e347bcc182650ec580563937ec41371c42872ae1ce8f072296",
    "05d77c057b19fcadf0952e3b87e90def3644aaa82a11d87f4ff47f739583db8b",
}
EXPECTED_SPLIT_PATIENTS = {"train": 5_940, "metacal": 661, "test": 1_649}
DELIVERY_LEGACY_FILES = (
    "manifest.json",
    "dataset_stats.json",
    "split.json",
    "split_stats.json",
    "cache_report.json",
    "frozen_baseline.json",
    "io_profile.json",
    "profiler_governance_on.txt",
    "profiler_governance_off.txt",
    "ablation_backbone.json",
)
DIRECT_DEPENDENCY_FILES = (
    "config.py",
    "image_quality.py",
    "build_cache.py",
    "validate_core_results.py",
)


@dataclass(frozen=True)
class StrictPaths:
    """严格 OOF 的隔离路径。"""

    exp: Path
    out: Path
    big: Path
    logs: Path
    markers: Path
    contract: Path
    validation_report: Path
    ready: Path
    delivery: Path


def paths(exp_dir: Path | None = None) -> StrictPaths:
    """返回隔离路径，不创建目录。"""
    exp = Path(exp_dir or os.environ.get("SR_EXP_DIR", legacy.EXP_DIR)).resolve()
    out = exp / "results" / PROTOCOL_ID
    big = exp / "big" / PROTOCOL_ID
    logs = exp / "logs" / PROTOCOL_ID
    return StrictPaths(
        exp=exp,
        out=out,
        big=big,
        logs=logs,
        markers=out / "markers",
        contract=out / "protocol_contract.json",
        validation_report=out / "validation_report.json",
        ready=out / "READY.json",
        delivery=out / "delivery",
    )


def ensure_runtime_dirs(p: StrictPaths) -> None:
    """仅在真实执行时创建隔离目录。"""
    for directory in (p.out, p.big, p.logs, p.markers):
        directory.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, value: str) -> None:
    """同目录原子发布文本。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    """同目录原子发布 JSON。"""
    atomic_write_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
    )


def canonical_json_sha256(value: Any) -> str:
    """计算可重现 JSON SHA-256。"""
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def cache_inventory_path(exp_dir: Path | None = None) -> Path:
    """返回严格协议私有的 JPEG 字节清单路径。"""
    return paths(exp_dir).out / "cache_inventory.json"


def build_cache_inventory(exp_dir: Path | None = None) -> dict[str, Any]:
    """逐项冻结 60,509 张实际训练 JPEG 的路径、大小和字节哈希。"""
    p = paths(exp_dir)
    train, metacal, test, _hashes = source_rows(exp_dir)
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in [*train, *metacal, *test]:
        target = _cache_path(p.exp / "big" / "jpeg384", row)
        relative = target.relative_to(p.exp).as_posix()
        if relative in seen:
            raise RuntimeError(f"duplicate cache path in inventory: {relative}")
        seen.add(relative)
        with Image.open(target) as image:
            if image.size != (IMAGE_SIZE, IMAGE_SIZE) or image.mode != "RGB":
                raise RuntimeError(
                    f"cache image contract mismatch: {relative} "
                    f"size={image.size} mode={image.mode}"
                )
            image.verify()
        entries.append(
            {
                "relative_path": relative,
                "size": target.stat().st_size,
                "sha256": file_sha256(target),
                "width": IMAGE_SIZE,
                "height": IMAGE_SIZE,
                "mode": "RGB",
            }
        )
    if len(entries) != EXPECTED_USABLE_IMAGES:
        raise RuntimeError(
            f"cache inventory count={len(entries)} != {EXPECTED_USABLE_IMAGES}"
        )
    payload = {
        "schema_version": 1,
        "count": len(entries),
        "entries_sha256": canonical_json_sha256(entries),
        "entries": entries,
    }
    atomic_write_json(cache_inventory_path(exp_dir), payload)
    return payload


def load_cache_inventory(
    exp_dir: Path | None = None, *, expected_file_sha256: str | None = None
) -> dict[str, dict[str, Any]]:
    """深验私有 JPEG 清单并按相对路径索引。"""
    target = cache_inventory_path(exp_dir)
    if expected_file_sha256 and file_sha256(target) != expected_file_sha256:
        raise RuntimeError("cache inventory file hash drift")
    payload = json.loads(target.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if (
        payload.get("schema_version") != 1
        or payload.get("count") != EXPECTED_USABLE_IMAGES
        or not isinstance(entries, list)
        or len(entries) != EXPECTED_USABLE_IMAGES
        or payload.get("entries_sha256") != canonical_json_sha256(entries)
    ):
        raise RuntimeError("cache inventory schema/count/self-hash invalid")
    indexed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        relative = entry.get("relative_path") if isinstance(entry, dict) else None
        if (
            not isinstance(relative, str)
            or relative in indexed
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or entry.get("width") != IMAGE_SIZE
            or entry.get("height") != IMAGE_SIZE
            or entry.get("mode") != "RGB"
            or not isinstance(entry.get("size"), int)
            or entry["size"] <= 0
            or not isinstance(entry.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
        ):
            raise RuntimeError("cache inventory contains invalid/duplicate entry")
        indexed[relative] = entry
    return indexed


def patient_of(biopsy_id: str) -> str:
    """从活检文件夹编号取患者键。"""
    return str(biopsy_id).split("_", 1)[0]


def _cache_path(cache_root: Path, row: dict[str, Any]) -> Path:
    name = Path(str(row["path"]).replace("\\", "/")).stem + ".jpg"
    return cache_root / row["cls"] / row["pt"] / name


def _normalized_source_path(value: str) -> str:
    return str(value).replace("\\", "/")


def _source_path_sha256(value: str) -> str:
    return hashlib.sha256(_normalized_source_path(value).encode("utf-8")).hexdigest()


def validated_qc_exclusions(exp_dir: Path | None = None) -> list[dict[str, str]]:
    """严格验证 60,512→60,509 的唯一 3 项 QC 白名单。"""
    p = paths(exp_dir)
    manifest = json.loads((p.exp / "results" / "manifest.json").read_text("utf-8"))
    split = json.loads((p.exp / "results" / "split.json").read_text("utf-8"))
    cache_report = json.loads(
        (p.exp / "results" / "cache_report.json").read_text("utf-8")
    )
    if len(manifest) != EXPECTED_RAW_IMAGES:
        raise RuntimeError(f"manifest raw images={len(manifest)} != 60512")
    if (
        int(cache_report.get("total", -1)) != EXPECTED_RAW_IMAGES
        or int(cache_report.get("converted", -1)) != EXPECTED_USABLE_IMAGES
        or len(cache_report.get("errors", [])) != 3
    ):
        raise RuntimeError("cache_report 不满足 60512/60509/3 契约")
    by_path: dict[str, list[dict[str, Any]]] = {}
    for row in manifest:
        by_path.setdefault(_normalized_source_path(row["path"]), []).append(row)
    exclusions = []
    for error in cache_report["errors"]:
        source = _normalized_source_path(error["src"])
        path_hash = _source_path_sha256(source)
        matches = by_path.get(source, [])
        if len(matches) != 1:
            raise RuntimeError(f"QC 异常项不能唯一映射 manifest: {path_hash}")
        row = matches[0]
        exclusions.append(
            {
                "cls": str(row["cls"]),
                "pt": str(row["pt"]),
                "subset": str(split[row["pt"]]),
                "source_path_sha256": path_hash,
                "error": str(error["err"]),
            }
        )
    if {row["source_path_sha256"] for row in exclusions} != EXPECTED_QC_PATH_SHA256:
        raise RuntimeError("QC 排除路径不是冻结的 3 项白名单")
    subset_counts = {
        subset: sum(row["subset"] == subset for row in exclusions)
        for subset in ("train", "metacal", "test")
    }
    if subset_counts != {"train": 2, "metacal": 0, "test": 1}:
        raise RuntimeError(f"QC 排除子集计数不匹配: {subset_counts}")
    return sorted(exclusions, key=lambda row: row["source_path_sha256"])


def source_rows(
    exp_dir: Path | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
]:
    """按 manifest 原始顺序返回可用 Train/Meta-Cal/Test 行。

    与旧流程一致，缓存中不存在的可重现损坏源图像在此排除。
    """
    p = paths(exp_dir)
    manifest_path = p.exp / "results" / "manifest.json"
    split_path = p.exp / "results" / "split.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    exclusions = validated_qc_exclusions(exp_dir)
    excluded_hashes = {row["source_path_sha256"] for row in exclusions}
    cache_root = p.exp / "big" / "jpeg384"
    train: list[dict[str, Any]] = []
    metacal: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for row in manifest:
        subset = split[row["pt"]]
        if subset not in {"train", "metacal", "test"}:
            raise ValueError(f"未知子集: {subset!r}")
        source_hash = _source_path_sha256(row["path"])
        if source_hash in excluded_hashes:
            continue
        if not _cache_path(cache_root, row).is_file():
            raise RuntimeError(
                "非白名单缓存缺失，失败关闭: " f"{row['cls']}/{row['pt']}/{source_hash}"
            )
        copied = {
            "cls": str(row["cls"]),
            "pt": str(row["pt"]),
            "path": str(row["path"]),
            "source_subset": subset,
        }
        {"train": train, "metacal": metacal, "test": test}[subset].append(copied)
    hashes = {
        "manifest_sha256": file_sha256(manifest_path),
        "split_sha256": file_sha256(split_path),
        "cache_report_sha256": file_sha256(p.exp / "results" / "cache_report.json"),
    }
    return train, metacal, test, hashes


def source_partition_rows(
    subset: str, exp_dir: Path | None = None
) -> list[dict[str, Any]]:
    """返回单一源分区；下游 ``load_rows`` 不接收或解码其他分区行。

    源 manifest 仍会被 JSON 解析后按 split 过滤，因此不声称未读取其他
    分区的原始 JSON 字节；协议保证的是它们不进入梯度、早停或选择。
    """
    if subset not in {"train", "metacal", "test"}:
        raise ValueError(f"unknown source partition: {subset!r}")
    p = paths(exp_dir)
    manifest = json.loads(
        (p.exp / "results" / "manifest.json").read_text(encoding="utf-8")
    )
    split = json.loads((p.exp / "results" / "split.json").read_text(encoding="utf-8"))
    excluded_hashes = {
        row["source_path_sha256"] for row in validated_qc_exclusions(exp_dir)
    }
    cache_root = p.exp / "big" / "jpeg384"
    selected: list[dict[str, Any]] = []
    for row in manifest:
        if split[row["pt"]] != subset:
            continue
        source_hash = _source_path_sha256(row["path"])
        if source_hash in excluded_hashes:
            continue
        if not _cache_path(cache_root, row).is_file():
            raise RuntimeError(
                "non-whitelisted cache missing, fail closed: "
                f"{row['cls']}/{row['pt']}/{source_hash}"
            )
        selected.append(
            {
                "cls": str(row["cls"]),
                "pt": str(row["pt"]),
                "path": str(row["path"]),
                "source_subset": subset,
            }
        )
    return selected


def rows_sha256(rows: Iterable[dict[str, Any]]) -> str:
    """对输入行的标识、类别和源子集绑定。"""
    compact = [
        {
            "cls": row["cls"],
            "pt": row["pt"],
            "path": str(row["path"]).replace("\\", "/"),
            "source_subset": row["source_subset"],
        }
        for row in rows
    ]
    return canonical_json_sha256(compact)


def _all_classes_present(labels: np.ndarray, indices: np.ndarray) -> bool:
    return set(np.unique(labels[indices]).tolist()) == set(range(NUM_CLASSES))


def grouped_assignments(
    rows: list[dict[str, Any]],
    outer_folds: int = OUTER_FOLDS,
    inner_folds: int = INNER_FOLDS,
    seed: int = SEED,
) -> tuple[dict[str, int], dict[str, dict[str, str]], list[dict[str, Any]]]:
    """生成外层 5 折与每个外折的 inner-validation 分配。

    标签只用于一次性、预先固定的分层分配。外层留出标签不会
    进入梯度、早停、epoch 选择或融合拟合。
    """
    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("外层和内层折数必须至少为 2")
    labels = np.asarray([legacy.CLASS_TO_IDX[row["cls"]] for row in rows])
    groups = np.asarray([patient_of(row["pt"]) for row in rows])
    if set(np.unique(labels)) != set(range(NUM_CLASSES)):
        raise RuntimeError("Train 未覆盖全部类别")
    splitter = StratifiedGroupKFold(
        n_splits=outer_folds,
        shuffle=True,
        random_state=seed,
    )
    outer_assignment: dict[str, int] = {}
    outer_splits: list[tuple[np.ndarray, np.ndarray]] = []
    records: list[dict[str, Any]] = []
    for fold, (train_idx, heldout_idx) in enumerate(
        splitter.split(np.zeros(len(labels)), labels, groups)
    ):
        train_groups = set(groups[train_idx])
        heldout_groups = set(groups[heldout_idx])
        if train_groups & heldout_groups:
            raise RuntimeError(f"外层第 {fold} 折患者泄漏")
        if not _all_classes_present(labels, train_idx):
            raise RuntimeError(f"外层第 {fold} 折训练部分缺类")
        if not _all_classes_present(labels, heldout_idx):
            raise RuntimeError(f"外层第 {fold} 折留出部分缺类")
        for group in heldout_groups:
            if group in outer_assignment:
                raise RuntimeError(f"患者 {group} 被分配多个外折")
            outer_assignment[str(group)] = fold
        outer_splits.append((train_idx, heldout_idx))
        records.append(
            {
                "fold": fold,
                "outer_train_images": int(len(train_idx)),
                "outer_heldout_images": int(len(heldout_idx)),
                "outer_train_patients": len(train_groups),
                "outer_heldout_patients": len(heldout_groups),
                "outer_patient_overlap": 0,
                "outer_train_all_classes": True,
                "outer_heldout_all_classes": True,
            }
        )
    if set(outer_assignment) != set(groups.tolist()):
        raise RuntimeError("外层分配未覆盖所有 Train 患者")

    inner_assignments: dict[str, dict[str, str]] = {}
    for outer_fold, (outer_train_idx, _heldout_idx) in enumerate(outer_splits):
        inner_y = labels[outer_train_idx]
        inner_groups = groups[outer_train_idx]
        inner_splitter = StratifiedGroupKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=seed + 10_000 + outer_fold,
        )
        candidates = list(
            inner_splitter.split(
                np.zeros(len(outer_train_idx)),
                inner_y,
                inner_groups,
            )
        )
        chosen_index = outer_fold % inner_folds
        inner_train_local, inner_valid_local = candidates[chosen_index]
        inner_train_idx = outer_train_idx[inner_train_local]
        inner_valid_idx = outer_train_idx[inner_valid_local]
        train_groups = set(groups[inner_train_idx])
        valid_groups = set(groups[inner_valid_idx])
        if train_groups & valid_groups:
            raise RuntimeError(f"外层第 {outer_fold} 折 inner 患者泄漏")
        if not _all_classes_present(labels, inner_train_idx):
            raise RuntimeError(f"外层第 {outer_fold} 折 inner-train 缺类")
        if not _all_classes_present(labels, inner_valid_idx):
            raise RuntimeError(f"外层第 {outer_fold} 折 inner-valid 缺类")
        mapping = {str(group): "train" for group in train_groups}
        mapping.update({str(group): "validation" for group in valid_groups})
        if set(mapping) != set(inner_groups.tolist()):
            raise RuntimeError(f"外层第 {outer_fold} 折 inner 未完整覆盖")
        inner_assignments[str(outer_fold)] = mapping
        records[outer_fold].update(
            {
                "inner_splitter": "StratifiedGroupKFold",
                "inner_n_splits": inner_folds,
                "inner_selected_fold": chosen_index,
                "inner_seed": seed + 10_000 + outer_fold,
                "inner_train_images": int(len(inner_train_idx)),
                "inner_validation_images": int(len(inner_valid_idx)),
                "inner_train_patients": len(train_groups),
                "inner_validation_patients": len(valid_groups),
                "inner_patient_overlap": 0,
                "inner_train_all_classes": True,
                "inner_validation_all_classes": True,
            }
        )
    return outer_assignment, inner_assignments, records


def source_code_sha256(exp_dir: Path | None = None) -> str:
    """绑定严格流程的所有 Python/Shell 源码。"""
    root = paths(exp_dir).exp / "strict_oof"
    entries = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item)):
        if path.is_file() and path.suffix in {".py", ".sh", ".service", ".timer"}:
            entries.append(
                {
                    "name": str(path.relative_to(root)).replace("\\", "/"),
                    "sha256": file_sha256(path),
                }
            )
    return canonical_json_sha256(entries)


def direct_dependency_sha256(exp_dir: Path | None = None) -> dict[str, str]:
    """绑定严格流程直接读取的命名空间外源码，防止长跑期间静默漂移。"""
    root = paths(exp_dir).exp
    result: dict[str, str] = {}
    for name in DIRECT_DEPENDENCY_FILES:
        target = root / name
        if not target.is_file():
            raise RuntimeError(f"strict OOF direct dependency missing: {name}")
        result[name] = file_sha256(target)
    return result


def validate_image_quality_v2(exp_dir: Path | None = None) -> dict[str, Any]:
    """校验启动前正式 image-quality v2 门。"""
    p = paths(exp_dir)
    target = p.out / "image_quality.json"
    value = json.loads(target.read_text(encoding="utf-8"))
    expected_by_class = {name: 30 for name in CLASSES}
    sampling = value.get("sampling", {})
    cache_audit = value.get("cache_byte_audit", {})
    if (
        value.get("schema_version") != 2
        or value.get("protocol", {}).get("per_class_requested") != 30
        or sampling.get("attempted") != 30 * NUM_CLASSES
        or sampling.get("succeeded") != 30 * NUM_CLASSES
        or sampling.get("failed") != 0
        or sampling.get("selected_by_class") != expected_by_class
        or sampling.get("successful_by_class") != expected_by_class
        or cache_audit.get("matches") != 30 * NUM_CLASSES
        or cache_audit.get("mismatches") != 0
        or cache_audit.get("all_match") is not True
    ):
        raise RuntimeError("image_quality.json 未通过正式 v2 每类30张硬门")
    footer = value.get("protocol", {}).get("footer_mask", {})
    if footer.get("rows") != 28 or footer.get("target_height") != 384:
        raise RuntimeError("image-quality v2 标注条遮罩口径不匹配")
    protocol = value.get("protocol", {})
    if (
        protocol.get("sample_seed") != SEED
        or protocol.get("color_space") != "RGB"
        or protocol.get("resize", {}).get("output") != [IMAGE_SIZE, IMAGE_SIZE]
        or protocol.get("resize", {}).get("resampler") != "Pillow Lanczos"
        or protocol.get("resize", {}).get("crop") is not False
        or protocol.get("jpeg", {}).get("quality") != JPEG_QUALITY
        or not math.isclose(
            float(footer.get("configured_area_fraction", -1)),
            28 / IMAGE_SIZE,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("image-quality v2 preprocessing protocol mismatch")
    manifest = json.loads(
        (p.exp / "results" / "manifest.json").read_text(encoding="utf-8")
    )
    rng = random.Random(SEED)
    by_class = {name: [] for name in CLASSES}
    for row in manifest:
        suffix = Path(str(row.get("path", ""))).suffix.lower()
        if row.get("cls") in by_class and suffix in {".tif", ".tiff"}:
            by_class[str(row["cls"])].append(row)
    digest = hashlib.sha256()
    for class_name in CLASSES:
        candidates = sorted(by_class[class_name], key=lambda row: str(row["path"]))
        if len(candidates) < 30:
            raise RuntimeError(
                f"image-quality class has fewer than 30 TIFFs: {class_name}"
            )
        for row in rng.sample(candidates, 30):
            digest.update(f"{class_name}\t{row['path']}\n".encode("utf-8"))
    if value.get("sampling", {}).get("ordered_sample_sha256") != digest.hexdigest():
        raise RuntimeError("image-quality deterministic sample hash mismatch")
    stages = {
        "resize_only_vs_native",
        "resize_mask_vs_native",
        "network_input_vs_native",
        "jpeg_vs_resize_mask",
    }
    metrics = {"ssim", "psnr_db", "lpips"}

    def validate_summaries(current: Any, expected_n: int, label: str) -> None:
        if not isinstance(current, dict) or set(current) != stages:
            raise RuntimeError(f"image-quality stages invalid: {label}")
        for stage, stage_metrics in current.items():
            if not isinstance(stage_metrics, dict) or set(stage_metrics) != metrics:
                raise RuntimeError(f"image-quality metrics invalid: {label}/{stage}")
            for metric, summary in stage_metrics.items():
                if not isinstance(summary, dict):
                    raise RuntimeError(
                        f"image-quality summary missing: {label}/{stage}/{metric}"
                    )
                numeric = [
                    summary.get(key) for key in ("mean", "sd", "median", "q025", "q975")
                ]
                if (
                    summary.get("n") != expected_n
                    or summary.get("n_nonfinite") != 0
                    or not all(
                        isinstance(item, (int, float)) and math.isfinite(float(item))
                        for item in numeric
                    )
                    or float(summary["sd"]) < 0
                    or summary.get("n_finite") != expected_n
                    or float(summary["q025"]) > float(summary["q975"])
                ):
                    raise RuntimeError(
                        f"image-quality summary invalid: {label}/{stage}/{metric}"
                    )
                low, high = {
                    "ssim": (-1.0, 1.0),
                    "psnr_db": (0.0, 200.0),
                    "lpips": (0.0, 2.0),
                }[metric]
                if not low <= float(summary["mean"]) <= high:
                    raise RuntimeError(
                        f"image-quality metric range invalid: {label}/{stage}/{metric}"
                    )

    validate_summaries(value.get("overall"), 30 * NUM_CLASSES, "overall")
    per_class = value.get("per_class")
    if not isinstance(per_class, dict) or set(per_class) != set(CLASSES):
        raise RuntimeError("image-quality per-class set mismatch")
    for class_name, summaries in per_class.items():
        validate_summaries(summaries, 30, class_name)
    return value


def _legacy_tree_sha256(p: StrictPaths) -> dict[str, str]:
    """绑定 prepare 时已有的全部旧 results 文件。"""
    root = p.exp / "results"
    result = {}
    for target in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not target.is_file():
            continue
        relative = target.relative_to(root)
        if relative.parts and relative.parts[0] == PROTOCOL_ID:
            continue
        if relative.as_posix() == "strict_oof_ACTIVE.json":
            continue
        result[relative.as_posix()] = file_sha256(target)
    return result


def static_protocol() -> dict[str, Any]:
    """返回与数据无关的预注册协议。"""
    hparams = dict(legacy.HPARAMS)
    return {
        "protocol_id": PROTOCOL_ID,
        "purpose": "Scientific Reports reviewer R4.1 strict backbone-level OOF",
        "train_definition": "original Train only; source split remains frozen",
        "metacal_definition": (
            "original Meta-Cal; independent fusion selection, calibration, "
            "and referral-threshold fitting"
        ),
        "terminal_test_definition": "original Test; labels sealed from all selection",
        "classes": list(CLASSES),
        "backbones": list(BACKBONES),
        "same_protocol_supervised_models": list(TRAIN_MODELS),
        "model_architectures": dict(MODEL_ARCHITECTURES),
        "ensemble_members": list(BACKBONES),
        "resnet50_role": (
            "same-protocol supervised standalone comparator; excluded from "
            "ensemble fitting and selection"
        ),
        "excluded_from_retraining": ["legacy_ablations", "optional_seeds"],
        "outer_splitter": "StratifiedGroupKFold",
        "outer_folds": OUTER_FOLDS,
        "inner_splitter": "StratifiedGroupKFold",
        "inner_folds": INNER_FOLDS,
        "inner_selection_rule": INNER_SELECTION_RULE,
        "seed": SEED,
        "epoch_selection": {
            "outer": "inner-validation macro-F1 with patience 8; outer heldout excluded",
            "terminal": TERMINAL_EPOCH_RULE,
            "terminal_early_stopping": False,
        },
        "scheduler": {
            "type": "linear warmup plus cosine decay",
            "horizon_epochs": SCHEDULER_HORIZON_EPOCHS,
            "terminal_behavior": "run is truncated at fixed epoch; horizon is not rescaled",
        },
        "hparams": hparams,
        "image_pipeline": {
            "img_size": IMAGE_SIZE,
            "jpeg_quality": JPEG_QUALITY,
            "cache_subdirectory": "big/jpeg384",
        },
        "training_recipe": {
            "sampler": True,
            "focal": True,
            "mixup_cutmix": True,
            "class_balanced_reweight": False,
        },
        "oof_tta_count": OOF_TTA_COUNT,
        "final_metacal_test_tta_count": FINAL_TTA_COUNT,
        "fusion_candidates": ["voting_tta8", "stack_lr_tta8"],
        "fusion_fit": "all original-Train backbone OOF features",
        "fusion_selection": (
            "fixed voting_tta8 versus stack_lr_tta8 on independent Meta-Cal"
        ),
        "temperature": "fit only on selected Meta-Cal probabilities; then frozen for Test",
        "referral_thresholds": list(legacy.REFERRAL_THRESHOLDS),
        "referral_rule": "lowest Meta-Cal threshold with autonomous accuracy >= 0.90",
        "additional_ablation_arms": {
            "cb_only": {
                "sampler": False,
                "focal": False,
                "mix_mode": "none",
                "cb_reweight": True,
            },
            "mixup_only": {
                "sampler": False,
                "focal": False,
                "mix_mode": "mixup",
                "cb_reweight": False,
            },
            "cutmix_only": {
                "sampler": False,
                "focal": False,
                "mix_mode": "cutmix",
                "cb_reweight": False,
            },
        },
        "additional_ablation_protocol": (
            "selected legacy ablation backbone; original Train fitting, "
            "original Meta-Cal early stopping, 30-epoch budget, isolated outputs"
        ),
        "additional_ablation_epoch_budget": ABLATION_EPOCH_BUDGET,
        "old_artifacts_overwritten": False,
    }


def build_contract(exp_dir: Path | None = None) -> dict[str, Any]:
    """从当前输入构造完整且确定的协议合同。"""
    p = paths(exp_dir)
    validate_image_quality_v2(exp_dir)
    preflight_path = p.out / "runtime_preflight.json"
    runtime_preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    from strict_oof.preflight import environment_lock, validate_runtime_preflight

    preflight_errors = validate_runtime_preflight(
        runtime_preflight,
        expected_source_sha256=source_code_sha256(exp_dir),
        expected_environment=environment_lock(),
    )
    if preflight_errors:
        raise RuntimeError(
            "runtime preflight missing, stale, or incomplete: " f"{preflight_errors}"
        )
    inventory_path = cache_inventory_path(exp_dir)
    inventory = load_cache_inventory(exp_dir)
    train, metacal, test, input_hashes = source_rows(exp_dir)
    if set(inventory) != {
        _cache_path(p.exp / "big" / "jpeg384", row).relative_to(p.exp).as_posix()
        for row in [*train, *metacal, *test]
    }:
        raise RuntimeError("cache inventory paths do not equal the 60,509 usable rows")
    outer, inner, folds = grouped_assignments(train)
    ablation_backbone_path = p.exp / "results" / "ablation_backbone.json"
    legacy_ablation_path = p.exp / "results" / "ablation_analysis.json"
    protected_legacy_paths = [
        p.exp / "results" / "analysis.json",
        p.exp / "results" / "core_validation.json",
        p.exp / "results" / "markers" / "ALL.done",
        p.exp / "results" / "markers" / "core_validation.done",
    ]
    ablation_backbone_record = json.loads(
        ablation_backbone_path.read_text(encoding="utf-8")
    )
    ablation_backbone = str(ablation_backbone_record["backbone"])
    if ablation_backbone not in BACKBONES:
        raise RuntimeError(f"非法消融骨干: {ablation_backbone}")
    legacy_ablation_arms = [
        key
        for key in json.loads(legacy_ablation_path.read_text(encoding="utf-8"))
        if not key.startswith("_")
    ]
    legacy_ablation_logs = {
        f"train_log_{ablation_backbone}_abl_{arm}.json": file_sha256(
            p.exp / "results" / f"train_log_{ablation_backbone}_abl_{arm}.json"
        )
        for arm in legacy_ablation_arms
    }
    patient_sets = {
        "train": {patient_of(row["pt"]) for row in train},
        "metacal": {patient_of(row["pt"]) for row in metacal},
        "test": {patient_of(row["pt"]) for row in test},
    }
    patient_counts = {name: len(values) for name, values in patient_sets.items()}
    if patient_counts != EXPECTED_SPLIT_PATIENTS:
        raise RuntimeError(f"患者划分硬门不匹配: {patient_counts}")
    if sum(patient_counts.values()) != 8_250:
        raise RuntimeError("患者总数闭合失败: 8250 != 5940+661+1649")
    if any(
        patient_sets[left] & patient_sets[right]
        for left, right in (
            ("train", "metacal"),
            ("train", "test"),
            ("metacal", "test"),
        )
    ):
        raise RuntimeError("Train/Meta-Cal/Test 患者存在交集")
    contract: dict[str, Any] = {
        **static_protocol(),
        "input_hashes": input_hashes,
        "train_rows_sha256": rows_sha256(train),
        "metacal_rows_sha256": rows_sha256(metacal),
        "test_rows_sha256": rows_sha256(test),
        "train_images": len(train),
        "metacal_images": len(metacal),
        "test_images": len(test),
        "train_patients": len({patient_of(row["pt"]) for row in train}),
        "metacal_patients": len({patient_of(row["pt"]) for row in metacal}),
        "test_patients": len({patient_of(row["pt"]) for row in test}),
        "patient_split_gate": {
            "total": 8_250,
            "by_subset": patient_counts,
            "pairwise_overlaps": {
                "train_metacal": 0,
                "train_test": 0,
                "metacal_test": 0,
            },
        },
        "outer_assignment": outer,
        "inner_assignments": inner,
        "fold_records": folds,
        "qc_contract": {
            "raw_images": EXPECTED_RAW_IMAGES,
            "usable_images": EXPECTED_USABLE_IMAGES,
            "excluded_images": 3,
            "by_subset": {"train": 2, "metacal": 0, "test": 1},
            "exclusions": validated_qc_exclusions(exp_dir),
        },
        "additional_ablation_backbone": ablation_backbone,
        "additional_ablation_backbone_record_sha256": file_sha256(
            ablation_backbone_path
        ),
        "legacy_ablation_analysis_sha256": file_sha256(legacy_ablation_path),
        "legacy_ablation_train_log_sha256": legacy_ablation_logs,
        "legacy_delivery_sha256": {
            name: file_sha256(p.exp / "results" / name)
            for name in DELIVERY_LEGACY_FILES
        },
        "strict_image_quality_sha256": file_sha256(p.out / "image_quality.json"),
        "protected_legacy_tree_sha256": _legacy_tree_sha256(p),
        "protected_legacy_sha256": {
            str(path.relative_to(p.exp)).replace("\\", "/"): file_sha256(path)
            for path in protected_legacy_paths
        },
        "source_code_sha256": source_code_sha256(exp_dir),
        "direct_dependency_sha256": direct_dependency_sha256(exp_dir),
        "runtime_preflight": runtime_preflight,
        "runtime_preflight_sha256": file_sha256(preflight_path),
        "cache_inventory_file": "results/strict_oof_v1/cache_inventory.json",
        "cache_inventory_file_sha256": file_sha256(inventory_path),
        "cache_inventory_entries_sha256": canonical_json_sha256(
            list(inventory.values())
        ),
        "cache_inventory_count": len(inventory),
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return contract


def verify_contract(contract: dict[str, Any], exp_dir: Path | None = None) -> None:
    """对协议自哈希、源码和输入双重失败关闭。"""
    expected_hash = contract.get("contract_sha256")
    body = dict(contract)
    body.pop("contract_sha256", None)
    if expected_hash != canonical_json_sha256(body):
        raise RuntimeError("协议 contract_sha256 不匹配")
    current = build_contract(exp_dir)
    if current["contract_sha256"] != expected_hash:
        differing = sorted(
            key
            for key in set(contract) | set(current)
            if contract.get(key) != current.get(key)
        )
        raise RuntimeError(f"协议已漂移，差异键: {differing}")


def load_contract(exp_dir: Path | None = None, fresh: bool = True) -> dict[str, Any]:
    """读取并验证已发布协议。"""
    p = paths(exp_dir)
    contract = json.loads(p.contract.read_text(encoding="utf-8"))
    body = dict(contract)
    stored = body.pop("contract_sha256", None)
    if stored != canonical_json_sha256(body):
        raise RuntimeError("协议文件自哈希无效")
    if fresh:
        verify_contract(contract, exp_dir)
    return contract


def partition_rows(
    rows: list[dict[str, Any]],
    contract: dict[str, Any],
    stage: str,
    fold: int | None = None,
) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    """按预先固定的患者分配返回训练/验证/留出行。"""
    indexed = list(enumerate(rows))
    if stage == "terminal":
        return {"train": indexed}
    if stage != "outer" or fold is None or not 0 <= fold < OUTER_FOLDS:
        raise ValueError("外层阶段必须提供 0..4 的 fold")
    inner = contract["inner_assignments"][str(fold)]
    result = {"train": [], "validation": [], "heldout": []}
    for index, row in indexed:
        patient = patient_of(row["pt"])
        if contract["outer_assignment"][patient] == fold:
            result["heldout"].append((index, row))
        else:
            result[inner[patient]].append((index, row))
    groups = {
        key: {patient_of(row["pt"]) for _index, row in values}
        for key, values in result.items()
    }
    if groups["train"] & groups["validation"]:
        raise RuntimeError("inner train/validation 患者重叠")
    if groups["heldout"] & (groups["train"] | groups["validation"]):
        raise RuntimeError("外层留出患者进入训练或早停")
    return result


def terminal_epoch(best_epochs: Iterable[int]) -> int:
    """按预先固定规则取 5 个外折最佳 epoch 的中位数。"""
    values = sorted(int(value) for value in best_epochs)
    if len(values) != OUTER_FOLDS:
        raise ValueError(f"必须提供 {OUTER_FOLDS} 个外折 best epoch")
    if any(value < 1 or value > SCHEDULER_HORIZON_EPOCHS for value in values):
        raise ValueError("best epoch 超出 1..40")
    return int(math.ceil(float(np.median(values))))


def scheduler_total_steps(steps_per_epoch: int) -> int:
    """调度器永远保持原 40 epoch 节奏。"""
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch 必须为正整数")
    return SCHEDULER_HORIZON_EPOCHS * steps_per_epoch
