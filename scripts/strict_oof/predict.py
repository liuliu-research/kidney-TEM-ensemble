"""严格 OOF 分片可续跑 TTA 推理。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from strict_oof.data import IndexedRamDataset, load_rows
from strict_oof.protocol import (
    FINAL_TTA_COUNT,
    NUM_CLASSES,
    OOF_TTA_COUNT,
    OUTER_FOLDS,
    TRAIN_MODELS,
    atomic_write_json,
    file_sha256,
    load_contract,
    partition_rows,
    paths,
    source_rows,
)
from strict_oof.train import artifact_paths, exclusive_lock

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def tta_batch(images: torch.Tensor, transform_index: int) -> torch.Tensor:
    """应用与原流程一致的确定 TTA 前缀。"""
    base = transform_index % 8
    if base in (2, 3):
        images = torch.rot90(images, 2, (2, 3))
    elif base in (4, 5):
        images = torch.rot90(images, 1, (2, 3))
    elif base in (6, 7):
        images = torch.rot90(images, 3, (2, 3))
    if base % 2 == 1:
        images = torch.flip(images, (3,))
    if transform_index >= 8:
        mean = _MEAN.to(images.device)
        std = _STD.to(images.device)
        pixels = (images * std + mean - 0.5) * 0.95 + 0.5
        images = (pixels.clamp(0, 1) - mean) / std
    return images


def prediction_paths(
    stage: str,
    model: str,
    subset: str,
    fold: int | None = None,
    arm: str | None = None,
) -> tuple[Path, Path, Path, Path]:
    """返回 bundle/shards/meta/lock 路径。"""
    p = paths()
    if stage == "outer":
        if fold is None:
            raise ValueError("outer 缺少 fold")
        stem = f"outer_fold{fold}_{model}"
        bundle = p.big / "predictions" / "outer" / f"fold_{fold}" / f"{model}.npz"
    elif stage == "terminal":
        stem = f"terminal_{model}_{subset}"
        bundle = p.big / "predictions" / "terminal" / f"{model}__{subset}.npz"
    elif stage == "ablation":
        if arm is None:
            raise ValueError("ablation 缺少 arm")
        stem = f"ablation_{arm}_{subset}"
        bundle = p.big / "predictions" / "ablation" / f"{arm}__{subset}.npz"
    else:
        raise ValueError(f"未知 stage: {stage}")
    shards = bundle.with_suffix("")
    meta = p.out / "predict" / f"{stem}.json"
    lock = p.big / "locks" / f"predict_{stem}.lock"
    return bundle, shards, meta, lock


def _load_prediction_rows(
    stage: str,
    subset: str,
    fold: int | None,
    contract: dict[str, Any],
) -> tuple[list[tuple[int, dict[str, Any]]], str]:
    train_rows, metacal_rows, test_rows, _hashes = source_rows()
    if stage == "outer":
        partitions = partition_rows(train_rows, contract, "outer", fold)
        return partitions["heldout"], "train_outer_heldout"
    if subset == "metacal":
        return list(enumerate(metacal_rows)), "metacal"
    if subset == "test":
        return list(enumerate(test_rows)), "test"
    raise ValueError(f"未知子集: {subset}")


def _bundle_valid(
    path: Path,
    expected_tta: int,
    expected_n: int,
    contract_hash: str,
    checkpoint_hash: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            probabilities = archive["probs"].astype(np.float32)
            seconds = archive["tta_seconds"].astype(np.float64)
            return bool(
                archive["probs"].shape == (expected_tta, expected_n, NUM_CLASSES)
                and archive["labels"].shape == (expected_n,)
                and archive["patients"].shape == (expected_n,)
                and archive["global_indices"].shape == (expected_n,)
                and str(archive["contract_sha256"].item()) == contract_hash
                and str(archive["checkpoint_sha256"].item()) == checkpoint_hash
                and seconds.shape == (expected_tta,)
                and np.all(np.isfinite(seconds))
                and np.all(seconds > 0)
                and np.all(np.isfinite(probabilities))
                and np.all(probabilities >= -0.002)
                and np.all(probabilities <= 1.002)
                and np.allclose(probabilities.sum(axis=2), 1.0, atol=0.02, rtol=0)
            )
    except (OSError, ValueError, KeyError):
        return False


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _shard_valid(
    path: Path,
    *,
    expected_n: int,
    contract_hash: str,
    checkpoint_hash: str,
    transform_index: int,
) -> tuple[bool, float | None]:
    """深验一个 TTA 分片；只让该损坏分片在生产者内被覆盖重算。"""
    if not path.is_file():
        return False, None
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "probs",
                "contract_sha256",
                "checkpoint_sha256",
                "transform_index",
                "seconds",
            }
            if not required.issubset(archive.files):
                return False, None
            probabilities = archive["probs"].astype(np.float32)
            seconds = float(archive["seconds"].item())
            valid = bool(
                probabilities.shape == (expected_n, NUM_CLASSES)
                and str(archive["contract_sha256"].item()) == contract_hash
                and str(archive["checkpoint_sha256"].item()) == checkpoint_hash
                and int(archive["transform_index"].item()) == transform_index
                and np.all(np.isfinite(probabilities))
                and np.all(probabilities >= -0.002)
                and np.all(probabilities <= 1.002)
                and np.allclose(probabilities.sum(axis=1), 1.0, atol=0.02, rtol=0)
                and np.isfinite(seconds)
                and seconds > 0
            )
            return (valid, seconds if valid else None)
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        return False, None


@torch.no_grad()
def run_prediction(
    stage: str,
    model_key: str,
    subset: str,
    fold: int | None,
    arm: str | None,
    batch_size: int,
) -> None:
    """逐 TTA 原子落盘，掉电后仅补未完成分片。"""
    contract = load_contract()
    from strict_oof.preflight import assert_frozen_runtime

    assert_frozen_runtime(contract)
    if stage == "ablation":
        model_key = contract["additional_ablation_backbone"]
    indexed_rows, source_partition = _load_prediction_rows(
        stage, subset, fold, contract
    )
    tta_count = OOF_TTA_COUNT if stage == "outer" else FINAL_TTA_COUNT
    bundle, shard_dir, meta_path, lock_path = prediction_paths(
        stage, model_key, subset, fold=fold, arm=arm
    )
    checkpoint, _resume, _train_log, _train_lock = artifact_paths(
        stage, model_key, fold=fold, arm=arm
    )
    checkpoint_hash = file_sha256(checkpoint)
    if _bundle_valid(
        bundle,
        tta_count,
        len(indexed_rows),
        contract["contract_sha256"],
        checkpoint_hash,
    ):
        print(f"{bundle.name} complete, skip", flush=True)
        return
    with exclusive_lock(lock_path):
        if _bundle_valid(
            bundle,
            tta_count,
            len(indexed_rows),
            contract["contract_sha256"],
            checkpoint_hash,
        ):
            return
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("contract_sha256") != contract["contract_sha256"]:
            raise RuntimeError("checkpoint 合同不匹配")
        expected_model_name = str(contract["model_architectures"][model_key])
        if state.get("model_name") != expected_model_name:
            raise RuntimeError("checkpoint 架构不匹配合同")
        import timm

        model = timm.create_model(
            expected_model_name, pretrained=False, num_classes=NUM_CLASSES
        )
        model.load_state_dict(state["model"], strict=True)
        model.eval().to("cuda")
        images, labels, global_indices, patients, _rows = load_rows(indexed_rows)
        dataset = IndexedRamDataset(
            images,
            labels,
            np.arange(len(labels), dtype=np.int64),
            train=False,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size, shuffle=False, num_workers=0
        )
        shard_dir.mkdir(parents=True, exist_ok=True)
        seconds: list[float] = []
        for transform_index in range(tta_count):
            shard = shard_dir / f"tta_{transform_index:02d}.npz"
            valid_shard, shard_seconds = _shard_valid(
                shard,
                expected_n=len(labels),
                contract_hash=contract["contract_sha256"],
                checkpoint_hash=checkpoint_hash,
                transform_index=transform_index,
            )
            if valid_shard:
                assert shard_seconds is not None
                seconds.append(shard_seconds)
                print(f"TTA {transform_index} shard valid, skip", flush=True)
                continue
            started = time.time()
            probabilities = np.empty((len(labels), NUM_CLASSES), dtype=np.float16)
            offset = 0
            for batch_images, _batch_labels in loader:
                batch_images = tta_batch(
                    batch_images.to("cuda", non_blocking=True),
                    transform_index,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch_images)
                current = torch.softmax(logits.float(), 1).cpu().numpy()
                probabilities[offset : offset + len(current)] = current.astype(
                    np.float16
                )
                offset += len(current)
            elapsed = time.time() - started
            _atomic_savez(
                shard,
                probs=probabilities,
                contract_sha256=np.asarray(contract["contract_sha256"]),
                checkpoint_sha256=np.asarray(checkpoint_hash),
                transform_index=np.asarray(transform_index, dtype=np.int16),
                seconds=np.asarray(elapsed, dtype=np.float64),
            )
            seconds.append(elapsed)
            print(f"TTA {transform_index} complete ({elapsed:.0f}s)", flush=True)
        stacked = []
        for transform_index in range(tta_count):
            with np.load(
                shard_dir / f"tta_{transform_index:02d}.npz", allow_pickle=False
            ) as archive:
                stacked.append(archive["probs"])
        probabilities = np.stack(stacked)
        row_sums = probabilities.astype(np.float32).sum(axis=2)
        if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0):
            raise RuntimeError("TTA 概率非法")
        _atomic_savez(
            bundle,
            probs=probabilities,
            labels=labels,
            patients=patients.astype(str),
            global_indices=global_indices,
            tta_seconds=np.asarray(seconds, dtype=np.float64),
            contract_sha256=np.asarray(contract["contract_sha256"]),
            checkpoint_sha256=np.asarray(checkpoint_hash),
            source_partition=np.asarray(source_partition),
            labels_used_for_model_or_selection=np.asarray(False),
        )
        metadata = {
            "status": "complete",
            "contract_sha256": contract["contract_sha256"],
            "checkpoint_sha256": checkpoint_hash,
            "stage": stage,
            "model": model_key,
            "fold": fold,
            "arm": arm,
            "subset": subset,
            "source_partition": source_partition,
            "n_images": len(labels),
            "tta_count": tta_count,
            "tta_seconds": seconds,
            "labels_used_for_model_or_selection": False,
            "bundle": str(bundle),
        }
        atomic_write_json(meta_path, metadata)
        print(f"saved {bundle}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["outer", "terminal", "ablation"], required=True
    )
    parser.add_argument("--model", choices=list(TRAIN_MODELS))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--subset", choices=["metacal", "test"])
    parser.add_argument("--arm", choices=["cb_only", "mixup_only", "cutmix_only"])
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    contract = load_contract()
    model = args.model
    subset = args.subset
    if args.stage == "outer":
        if model is None or args.fold not in range(OUTER_FOLDS):
            parser.error("outer 需 --model 和 --fold 0..4")
        subset = "train_oof"
    elif args.stage == "terminal":
        if model is None or subset not in {"metacal", "test"}:
            parser.error("terminal 需 --model 与 --subset")
    else:
        if args.arm is None or subset not in {"metacal", "test"}:
            parser.error("ablation 需 --arm 与 --subset")
        model = contract["additional_ablation_backbone"]
    if args.batch < 1:
        parser.error("--batch 必须为正整数")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "model": model,
                    "fold": args.fold,
                    "subset": subset,
                    "arm": args.arm,
                    "contract_sha256": contract["contract_sha256"],
                },
                indent=2,
            )
        )
        return 0
    run_prediction(
        args.stage,
        str(model),
        str(subset),
        args.fold,
        args.arm,
        args.batch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
