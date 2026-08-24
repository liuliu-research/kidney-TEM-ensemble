"""严格 OOF 外折、终末骨干与缺失消融臂训练。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

import config as legacy
from strict_oof.data import IndexedRamDataset, load_rows
from strict_oof.protocol import (
    NUM_CLASSES,
    OUTER_FOLDS,
    SCHEDULER_HORIZON_EPOCHS,
    TRAIN_MODELS,
    atomic_write_json,
    load_contract,
    partition_rows,
    paths,
    scheduler_total_steps,
    source_partition_rows,
    terminal_epoch,
)

STRICT_RECIPE = {
    "sampler": True,
    "focal": True,
    "mix_mode": "alternating",
    "cb_reweight": False,
}


class GpuMonitor(threading.Thread):
    """1 Hz GPU 利用率采样。"""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.samples: list[float] = []
        self._should_stop = False

    def run(self) -> None:
        while not self._should_stop:
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.samples.append(float(result.stdout.strip().splitlines()[0]))
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            time.sleep(1.0)

    def finish(self) -> list[float]:
        """停止并返回当前采样。"""
        self._should_stop = True
        self.join(timeout=3)
        return list(self.samples)


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """进程级互斥；掉电后内核自动释放。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    try:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except (BlockingIOError, OSError) as error:
        handle.close()
        raise RuntimeError(f"训练步骤已有存活进程: {path}") from error
    try:
        yield
    finally:
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            else:
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            handle.close()


def soft_target_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float,
    smoothing: float,
    alpha: torch.Tensor | None,
) -> torch.Tensor:
    """支持软标签的 Focal/CE 损失。"""
    n_classes = logits.size(1)
    if target.dim() == 1:
        soft = F.one_hot(target, n_classes).float()
    else:
        soft = target
    if smoothing > 0:
        soft = soft * (1 - smoothing) + smoothing / n_classes
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    weights = (1 - probs).pow(gamma) if gamma > 0 else torch.ones_like(probs)
    if alpha is not None:
        weights = weights * alpha.to(logits.device)[None, :]
    return -(soft * weights * log_probs).sum(1).mean()


def apply_mix(
    images: torch.Tensor,
    labels: torch.Tensor,
    mode: str,
    rng: np.random.Generator,
    hparams: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """按预注册模式应用 MixUp/CutMix。"""
    soft = F.one_hot(labels, NUM_CLASSES).float()
    if mode == "none" or rng.random() >= float(hparams["mix_prob"]):
        return images, soft
    selected = mode
    if mode == "alternating":
        selected = "mixup" if rng.random() < 0.5 else "cutmix"
    permutation = torch.randperm(images.size(0), device=images.device)
    if selected == "mixup":
        alpha = float(hparams["mixup_alpha"])
        lam = float(rng.beta(alpha, alpha))
        images = lam * images + (1 - lam) * images[permutation]
        soft = lam * soft + (1 - lam) * soft[permutation]
    elif selected == "cutmix":
        alpha = float(hparams["cutmix_alpha"])
        lam = float(rng.beta(alpha, alpha))
        height, width = images.shape[2:]
        rect_h = int(height * (1 - lam) ** 0.5)
        rect_w = int(width * (1 - lam) ** 0.5)
        center_y = int(rng.integers(height))
        center_x = int(rng.integers(width))
        y1 = max(center_y - rect_h // 2, 0)
        y2 = min(center_y + rect_h // 2, height)
        x1 = max(center_x - rect_w // 2, 0)
        x2 = min(center_x + rect_w // 2, width)
        images[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
        adjusted = 1 - (y2 - y1) * (x2 - x1) / (height * width)
        soft = adjusted * soft + (1 - adjusted) * soft[permutation]
    else:
        raise ValueError(f"未知 mix_mode: {mode}")
    return images, soft


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader) -> float:
    """在 inner-validation 或 Meta-Cal 上计算 macro-F1。"""
    model.eval()
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, batch_labels in loader:
        images = images.to("cuda", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite validation logits")
        predictions.append(logits.float().argmax(1).cpu().numpy())
        labels.append(np.asarray(batch_labels))
    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(labels)
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0,
        )
    )


def artifact_paths(
    stage: str,
    model: str,
    fold: int | None = None,
    arm: str | None = None,
) -> tuple[Path, Path, Path, Path]:
    """返回 checkpoint/resume/log/lock 路径。"""
    p = paths()
    if stage == "outer":
        if fold is None:
            raise ValueError("outer 缺少 fold")
        stem = f"outer_fold{fold}_{model}"
        checkpoint = p.big / "checkpoints" / "outer" / f"fold_{fold}" / f"{model}.pt"
    elif stage == "terminal":
        stem = f"terminal_{model}"
        checkpoint = p.big / "checkpoints" / "terminal" / f"{model}.pt"
    elif stage == "ablation":
        if arm is None:
            raise ValueError("ablation 缺少 arm")
        stem = f"ablation_{arm}"
        checkpoint = p.big / "checkpoints" / "ablation" / f"{arm}.pt"
    else:
        raise ValueError(f"未知 stage: {stage}")
    resume = checkpoint.with_name(checkpoint.stem + ".resume.pt")
    log = p.out / "train" / f"{stem}.json"
    lock = p.big / "locks" / f"{stem}.lock"
    return checkpoint, resume, log, lock


def atomic_torch_save(value: Any, path: Path) -> None:
    """原子发布 PyTorch 归档。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """复制一份不再随当前模型更新的 CPU 权重。"""
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def _state_dict_errors(model: torch.nn.Module, value: Any, label: str) -> list[str]:
    """不修改模型地检查 state_dict 的键、形状和 dtype。"""
    if not isinstance(value, dict) or not value:
        return [f"{label} missing/non-dict"]
    expected = model.state_dict()
    if set(value) != set(expected):
        return [f"{label} parameter keys mismatch"]
    errors: list[str] = []
    for key, tensor in value.items():
        reference = expected[key]
        if not isinstance(tensor, torch.Tensor):
            errors.append(f"{label}.{key} is not tensor")
        elif tensor.shape != reference.shape:
            errors.append(f"{label}.{key} shape mismatch")
        elif tensor.dtype != reference.dtype:
            errors.append(f"{label}.{key} dtype mismatch")
        elif (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(
            tensor
        ).all():
            errors.append(f"{label}.{key} contains non-finite values")
        if errors:
            break
    return errors


def _resume_errors(
    state: Any,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    contract_hash: str,
    run_key: str,
    model_name: str,
    max_epochs: int,
    early_stopping: bool,
    steps_per_epoch: int,
    expected_lrs: list[float],
) -> list[str]:
    """在恢复任何训练状态前深验 resume；错误必须回退到干净起点。"""
    if not isinstance(state, dict):
        return ["resume top-level is not dict"]
    errors: list[str] = []
    if state.get("contract_sha256") != contract_hash:
        errors.append("resume contract mismatch")
    if state.get("run_key") != run_key:
        errors.append("resume run_key mismatch")
    if state.get("model_name") != model_name:
        errors.append("resume model_name mismatch")
    try:
        epoch = int(state["epoch"])
        best_epoch = int(state["best_epoch"])
        wait = int(state["wait"])
    except (KeyError, TypeError, ValueError):
        return errors + ["resume epoch metadata invalid"]
    if not (1 <= epoch <= max_epochs):
        errors.append("resume epoch out of range")
    if not (1 <= best_epoch <= epoch):
        errors.append("resume best_epoch out of range")
    if wait < 0:
        errors.append("resume wait is negative")
    epochs = state.get("epochs")
    if (
        not isinstance(epochs, list)
        or len(epochs) != epoch
        or not epochs
        or not isinstance(epochs[-1], dict)
        or int(epochs[-1].get("epoch", -1)) != epoch
    ):
        errors.append("resume epoch history invalid")
    elif isinstance(epochs, list):
        for expected_epoch, row in enumerate(epochs, start=1):
            try:
                loss_value = float(row["loss"])
                seconds_value = float(row["epoch_seconds"])
                monitor_value = row.get("monitor_macro_f1")
                if (
                    int(row["epoch"]) != expected_epoch
                    or not np.isfinite(loss_value)
                    or loss_value < 0
                    or not np.isfinite(seconds_value)
                    or seconds_value <= 0
                    or (
                        early_stopping
                        and (
                            monitor_value is None
                            or not np.isfinite(float(monitor_value))
                        )
                    )
                    or (not early_stopping and monitor_value is not None)
                ):
                    raise ValueError
            except (KeyError, TypeError, ValueError, OverflowError):
                errors.append("resume epoch history row semantic invalid")
                break
    if (
        early_stopping
        and isinstance(epochs, list)
        and all(isinstance(row, dict) for row in epochs)
    ):
        recomputed_best = -1.0
        recomputed_epoch = -1
        recomputed_wait = 0
        for row in epochs:
            try:
                score = float(row["monitor_macro_f1"])
            except (KeyError, TypeError, ValueError):
                errors.append("resume monitor history invalid")
                break
            if not np.isfinite(score):
                errors.append("resume monitor history non-finite")
                break
            if score > recomputed_best:
                recomputed_best = score
                recomputed_epoch = int(row["epoch"])
                recomputed_wait = 0
            else:
                recomputed_wait += 1
        if (
            recomputed_epoch != best_epoch
            or recomputed_wait != wait
            or not np.isclose(
                recomputed_best,
                float(state.get("best_f1", np.nan)),
                rtol=0,
                atol=1e-15,
            )
        ):
            errors.append("resume early-stopping state != epoch history")
    elif not early_stopping and not (
        best_epoch == epoch
        and wait == 0
        and np.isnan(float(state.get("best_f1", np.nan)))
    ):
        errors.append("terminal resume fixed-epoch state invalid")
    errors.extend(_state_dict_errors(model, state.get("model"), "resume.model"))
    if early_stopping:
        errors.extend(
            _state_dict_errors(model, state.get("best_model"), "resume.best_model")
        )
    optimizer_state = state.get("optimizer")
    try:
        if not isinstance(optimizer_state, dict) or set(optimizer_state) != {
            "state",
            "param_groups",
        }:
            raise ValueError("top-level")
        saved_slots = optimizer_state["state"]
        saved_groups = optimizer_state["param_groups"]
        if not isinstance(saved_slots, dict) or not isinstance(saved_groups, list):
            raise ValueError("state/groups types")
        if len(saved_groups) != len(optimizer.param_groups):
            raise ValueError("group count")
        id_to_parameter: dict[int, torch.nn.Parameter] = {}
        all_ids: list[int] = []
        for group_index, (saved, current) in enumerate(
            zip(saved_groups, optimizer.param_groups)
        ):
            saved_ids = saved.get("params")
            current_params = current.get("params")
            if (
                not isinstance(saved_ids, list)
                or len(saved_ids) != len(current_params)
                or any(not isinstance(item, int) for item in saved_ids)
            ):
                raise ValueError("parameter ids")
            for parameter_id, parameter in zip(saved_ids, current_params):
                if parameter_id in id_to_parameter:
                    raise ValueError("duplicate parameter id")
                id_to_parameter[parameter_id] = parameter
                all_ids.append(parameter_id)
            if set(saved) != set(current):
                raise ValueError("parameter-group key set")
            for key in set(current) - {"params", "lr", "initial_lr"}:
                if saved[key] != current[key]:
                    raise ValueError(f"hyperparameter {key}")
            if not np.isclose(
                float(saved.get("lr")),
                expected_lrs[group_index],
                rtol=0,
                atol=1e-15,
            ):
                raise ValueError("learning rate")
            if not np.isclose(
                float(saved.get("initial_lr")),
                float(scheduler.base_lrs[group_index]),
                rtol=0,
                atol=1e-15,
            ):
                raise ValueError("initial learning rate")
        if len(all_ids) != len(set(all_ids)) or set(saved_slots) != set(all_ids):
            raise ValueError("optimizer slots do not cover parameters")
        for parameter_id, slot in saved_slots.items():
            parameter = id_to_parameter[parameter_id]
            if not isinstance(slot, dict) or set(slot) != {
                "step",
                "exp_avg",
                "exp_avg_sq",
            }:
                raise ValueError("AdamW slot keys")
            step_tensor = slot["step"]
            if (
                not isinstance(step_tensor, torch.Tensor)
                or step_tensor.numel() != 1
                or not torch.isfinite(step_tensor).all()
                or int(step_tensor.item()) != epoch * steps_per_epoch
            ):
                raise ValueError("AdamW step")
            for key in ("exp_avg", "exp_avg_sq"):
                tensor = slot[key]
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.shape != parameter.shape
                    or tensor.dtype != parameter.dtype
                    or not torch.isfinite(tensor).all()
                ):
                    raise ValueError(f"AdamW {key}")
    except (KeyError, TypeError, AttributeError, ValueError, OverflowError):
        errors.append("resume optimizer state invalid")
    scheduler_state = state.get("scheduler")
    fresh_scheduler_state = scheduler.state_dict()
    if not isinstance(scheduler_state, dict) or set(scheduler_state) != set(
        fresh_scheduler_state
    ):
        errors.append("resume scheduler state invalid")
    elif (
        int(scheduler_state.get("last_epoch", -1)) != epoch * steps_per_epoch
        or int(scheduler_state.get("_step_count", -1)) != epoch * steps_per_epoch + 1
        or not np.allclose(
            scheduler_state.get("base_lrs", []),
            fresh_scheduler_state.get("base_lrs", []),
            rtol=0,
            atol=1e-15,
        )
        or not np.allclose(
            scheduler_state.get("_last_lr", []),
            expected_lrs,
            rtol=0,
            atol=1e-15,
        )
    ):
        errors.append("resume scheduler step/LR state invalid")
    elif any(
        scheduler_state[key] != fresh_scheduler_state[key]
        for key in set(fresh_scheduler_state)
        - {"last_epoch", "_step_count", "_last_lr", "base_lrs"}
    ):
        errors.append("resume scheduler static semantics invalid")
    for key in ("mix_rng_state", "augment_rng_state"):
        if not isinstance(state.get(key), dict):
            errors.append(f"resume {key} invalid")
    for key in ("loader_rng_state", "torch_rng_state"):
        value = state.get(key)
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
            errors.append(f"resume {key} invalid")
    cuda_state = state.get("cuda_rng_state")
    if torch.cuda.is_available() and (
        not isinstance(cuda_state, torch.Tensor) or cuda_state.dtype != torch.uint8
    ):
        errors.append("resume cuda_rng_state invalid")
    try:
        best_f1 = float(state["best_f1"])
        elapsed = float(state.get("elapsed_seconds", 0.0))
        gpu = np.asarray(state.get("gpu_samples", []), dtype=float)
        if early_stopping and not np.isfinite(best_f1):
            errors.append("resume best_f1 is non-finite")
        if not np.isfinite(elapsed) or elapsed < 0:
            errors.append("resume elapsed_seconds invalid")
        if np.any(~np.isfinite(gpu)):
            errors.append("resume gpu_samples non-finite")
    except (KeyError, TypeError, ValueError):
        errors.append("resume scalar/history fields invalid")
    return errors


def _quarantine_resume(path: Path, reason: str) -> None:
    """只隔离损坏的本步 resume；不影响健康 checkpoint 或其他步骤。"""
    if not path.exists():
        return
    target = path.with_name(f"{path.name}.{datetime.now():%Y%m%d_%H%M%S_%f}.invalid")
    path.replace(target)
    print(f"quarantined invalid resume {path.name}: {reason}", flush=True)


def _already_complete(
    checkpoint: Path,
    log_path: Path,
    contract_hash: str,
    run_key: str,
) -> bool:
    if not checkpoint.is_file() or not log_path.is_file():
        return False
    try:
        record = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metadata_ok = (
        record.get("contract_sha256") == contract_hash
        and record.get("run_key") == run_key
        and record.get("status") == "complete"
        and checkpoint.stat().st_size > 1_000_000
    )
    if not metadata_ok:
        return False
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, EOFError):
        return False
    return (
        isinstance(state, dict)
        and state.get("contract_sha256") == contract_hash
        and state.get("run_key") == run_key
        and isinstance(state.get("model_name"), str)
        and state["model_name"] == record.get("model_name")
        and isinstance(state.get("model"), dict)
        and bool(state["model"])
        and int(state.get("epoch", -1)) == int(record.get("best_epoch", -2))
    )


def _terminal_epochs(model: str, contract_hash: str) -> tuple[int, list[int]]:
    values = []
    for fold in range(OUTER_FOLDS):
        _checkpoint, _resume, log_path, _lock = artifact_paths(
            "outer", model, fold=fold
        )
        record = json.loads(log_path.read_text(encoding="utf-8"))
        if record.get("contract_sha256") != contract_hash:
            raise RuntimeError(f"{model} fold {fold} 日志合同不匹配")
        values.append(int(record["best_epoch"]))
    return terminal_epoch(values), values


def _prepare_data(
    stage: str,
    contract: dict[str, Any],
    fold: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    train_rows = source_partition_rows("train")
    if stage == "outer":
        partitions = partition_rows(train_rows, contract, "outer", fold)
        allowed = partitions["train"] + partitions["validation"]
        images, labels, _global, _patients, _rows = load_rows(allowed)
        n_train = len(partitions["train"])
        train_indices = np.arange(n_train, dtype=np.int64)
        valid_indices = np.arange(n_train, len(allowed), dtype=np.int64)
        return images, labels, train_indices, valid_indices, "inner_validation"
    if stage == "terminal":
        allowed = list(enumerate(train_rows))
        images, labels, _global, _patients, _rows = load_rows(allowed)
        train_indices = np.arange(len(allowed), dtype=np.int64)
        return images, labels, train_indices, np.empty(0, dtype=np.int64), "none"
    metacal_rows = source_partition_rows("metacal")
    allowed_train = list(enumerate(train_rows))
    offset = len(train_rows)
    allowed_valid = [(offset + index, row) for index, row in enumerate(metacal_rows)]
    images, labels, _global, _patients, _rows = load_rows(allowed_train + allowed_valid)
    train_indices = np.arange(len(allowed_train), dtype=np.int64)
    valid_indices = np.arange(
        len(allowed_train), len(allowed_train) + len(allowed_valid), dtype=np.int64
    )
    return images, labels, train_indices, valid_indices, "metacal"


def run_training(
    stage: str,
    model_key: str,
    fold: int | None,
    arm: str | None,
) -> None:
    """执行一个可断点续跑的训练步骤。"""
    contract = load_contract()
    from strict_oof.preflight import (
        classifier_parameter_partition,
        environment_lock,
        model_state_sha256,
    )

    current_environment = environment_lock()
    if current_environment != contract["runtime_preflight"]["environment_lock"]:
        raise RuntimeError("runtime environment drifted from frozen preflight")
    if stage == "ablation":
        model_key = contract["additional_ablation_backbone"]
        recipe = dict(contract["additional_ablation_arms"][str(arm)])
        max_epochs = int(contract["additional_ablation_epoch_budget"])
        scheduler_horizon = max_epochs
        early_stopping = True
    else:
        recipe = dict(STRICT_RECIPE)
        scheduler_horizon = SCHEDULER_HORIZON_EPOCHS
        if stage == "outer":
            max_epochs = SCHEDULER_HORIZON_EPOCHS
            early_stopping = True
        else:
            max_epochs, fold_epochs = _terminal_epochs(
                model_key, contract["contract_sha256"]
            )
            early_stopping = False
    checkpoint, resume, log_path, lock_path = artifact_paths(
        stage, model_key, fold=fold, arm=arm
    )
    run_key = (
        f"{stage}:{model_key}:fold={fold}"
        if stage != "ablation"
        else f"ablation:{arm}:{model_key}"
    )
    if _already_complete(checkpoint, log_path, contract["contract_sha256"], run_key):
        print(f"{run_key} complete, skip", flush=True)
        return
    with exclusive_lock(lock_path):
        if _already_complete(
            checkpoint, log_path, contract["contract_sha256"], run_key
        ):
            return
        images, labels, train_indices, valid_indices, monitor_partition = _prepare_data(
            stage, contract, fold
        )
        seed = (
            int(contract["seed"])
            + TRAIN_MODELS.index(model_key) * 1_000
            + (fold or 0) * 100
            + {"outer": 0, "terminal": 50_000, "ablation": 80_000}[stage]
            + (list(contract["additional_ablation_arms"]).index(arm) if arm else 0)
        )
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        loader_generator = torch.Generator().manual_seed(seed)

        hp = dict(contract["hparams"])
        if "efficientnet" in model_key:
            hp["batch_size"] = hp["batch_size_effnet"]
        train_dataset = IndexedRamDataset(
            images, labels, train_indices, train=True, seed=seed
        )
        valid_dataset = (
            IndexedRamDataset(images, labels, valid_indices, train=False, seed=seed)
            if len(valid_indices)
            else None
        )
        counts = np.bincount(labels[train_indices], minlength=NUM_CLASSES)
        if np.any(counts == 0):
            raise RuntimeError("训练分区缺类，停止")
        if recipe["sampler"]:
            weights = (1.0 / counts)[labels[train_indices]]
            sampler = WeightedRandomSampler(
                torch.as_tensor(weights, dtype=torch.double),
                len(train_indices),
                generator=loader_generator,
            )
            train_loader = DataLoader(
                train_dataset,
                hp["batch_size"],
                sampler=sampler,
                num_workers=0,
                drop_last=True,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                hp["batch_size"],
                shuffle=True,
                generator=loader_generator,
                num_workers=0,
                drop_last=True,
            )
        valid_loader = (
            DataLoader(valid_dataset, 64, shuffle=False, num_workers=0)
            if valid_dataset is not None
            else None
        )

        import timm

        model_name = str(contract["model_architectures"][model_key])
        steps_per_epoch = len(train_loader)
        total_steps = (
            scheduler_total_steps(steps_per_epoch)
            if scheduler_horizon == SCHEDULER_HORIZON_EPOCHS
            else scheduler_horizon * steps_per_epoch
        )
        warmup_steps = int(total_steps * hp["warmup_frac"])

        def lr_multiplier(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))

        optimizer_group_metadata: dict[str, Any] = {}

        def new_training_objects() -> tuple[
            torch.nn.Module,
            torch.optim.Optimizer,
            torch.optim.lr_scheduler.LRScheduler,
        ]:
            """构建一套干净训练对象，供启动及坏 resume 回退。"""
            current_model = timm.create_model(
                model_name, pretrained=True, num_classes=NUM_CLASSES
            )
            current_pretrained_hash = model_state_sha256(current_model)
            expected_pretrained_hash = contract["runtime_preflight"][
                "pretrained_backbone_sha256"
            ][model_key]
            if current_pretrained_hash != expected_pretrained_hash:
                raise RuntimeError(
                    f"pretrained backbone drift for {model_key}: "
                    f"{current_pretrained_hash} != {expected_pretrained_hash}"
                )
            current_model.to("cuda")
            body_parameters, head_parameters, group_metadata = (
                classifier_parameter_partition(current_model)
            )
            expected_group_metadata = contract["runtime_preflight"][
                "pretrained_fingerprint_metadata"
            ][model_key]
            for key, observed in group_metadata.items():
                if observed != expected_group_metadata.get(key):
                    raise RuntimeError(
                        f"optimizer parameter partition drift for {model_key}/{key}: "
                        f"{observed} != {expected_group_metadata.get(key)}"
                    )
            nonlocal optimizer_group_metadata
            optimizer_group_metadata = group_metadata
            current_optimizer = torch.optim.AdamW(
                [
                    {
                        "params": body_parameters,
                        "lr": hp["base_lr"] * hp["backbone_lr_mult"],
                    },
                    {"params": head_parameters, "lr": hp["base_lr"]},
                ],
                weight_decay=hp["weight_decay"],
                betas=(0.9, 0.999),
            )
            current_scheduler = torch.optim.lr_scheduler.LambdaLR(
                current_optimizer, lr_multiplier
            )
            return current_model, current_optimizer, current_scheduler

        model, optimizer, scheduler = new_training_objects()
        alpha_tensor = None
        if recipe["cb_reweight"]:
            beta = 0.9999
            effective = (1 - np.power(beta, counts)) / (1 - beta)
            alpha = (1 / effective) * NUM_CLASSES / (1 / effective).sum()
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)
        gamma = hp["focal_gamma"] if recipe["focal"] else 0.0
        best_f1 = -1.0
        best_epoch = -1
        wait = 0
        epoch_rows: list[dict[str, Any]] = []
        elapsed_before = 0.0
        gpu_before: list[float] = []
        best_model_state: dict[str, torch.Tensor] | None = None
        start_epoch = 1
        if resume.exists():
            try:
                state = torch.load(resume, map_location="cpu", weights_only=False)
                resume_errors = _resume_errors(
                    state,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    contract_hash=contract["contract_sha256"],
                    run_key=run_key,
                    model_name=model_name,
                    max_epochs=max_epochs,
                    early_stopping=early_stopping,
                    steps_per_epoch=steps_per_epoch,
                    expected_lrs=(
                        [
                            float(base)
                            * float(
                                lr_multiplier(int(state["epoch"]) * steps_per_epoch)
                            )
                            for base in scheduler.base_lrs
                        ]
                        if isinstance(state, dict) and "epoch" in state
                        else []
                    ),
                )
                if resume_errors:
                    raise ValueError("; ".join(resume_errors))
                assert isinstance(state, dict)
                model.load_state_dict(state["model"], strict=True)
                model.to("cuda")
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                best_f1 = float(state["best_f1"])
                best_epoch = int(state["best_epoch"])
                wait = int(state["wait"])
                epoch_rows = list(state["epochs"])
                elapsed_before = float(state.get("elapsed_seconds", 0.0))
                gpu_before = list(state.get("gpu_samples", []))
                rng.bit_generator.state = state["mix_rng_state"]
                train_dataset.rng.bit_generator.state = state["augment_rng_state"]
                loader_generator.set_state(state["loader_rng_state"])
                torch.set_rng_state(state["torch_rng_state"])
                if (
                    torch.cuda.is_available()
                    and state.get("cuda_rng_state") is not None
                ):
                    torch.cuda.set_rng_state(state["cuda_rng_state"])
                elif torch.cuda.is_available():
                    raise ValueError("resume CUDA RNG state missing")
                if early_stopping:
                    best_model_state = dict(state["best_model"])
                    atomic_torch_save(
                        {
                            "model": best_model_state,
                            "model_name": model_name,
                            "contract_sha256": contract["contract_sha256"],
                            "run_key": run_key,
                            "epoch": best_epoch,
                            "monitor_macro_f1": best_f1,
                            "monitor_partition": monitor_partition,
                            "outer_heldout_used_for_training_or_selection": False,
                        },
                        checkpoint,
                    )
                start_epoch = int(state["epoch"]) + 1
                print(f"{run_key} resume at epoch {start_epoch}", flush=True)
                if early_stopping and wait >= hp["early_stop_patience"]:
                    # 断电可能发生在写完已达 patience 的 resume 后、写总结前。
                    # 此时必须直接收尾，不得再多训一轮。
                    start_epoch = max_epochs + 1
                    print(
                        f"{run_key} resume already reached early-stop patience; "
                        "finalize without another epoch",
                        flush=True,
                    )
            except (
                OSError,
                RuntimeError,
                ValueError,
                EOFError,
                KeyError,
                TypeError,
                AttributeError,
                OverflowError,
            ) as error:
                _quarantine_resume(resume, str(error))
                # 任一恢复动作失败后，模型/优化器/RNG可能已部分污染；全部重建。
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                rng = np.random.default_rng(seed)
                train_dataset.rng = np.random.default_rng(seed)
                loader_generator.manual_seed(seed)
                model, optimizer, scheduler = new_training_objects()
                best_f1 = -1.0
                best_epoch = -1
                wait = 0
                epoch_rows = []
                elapsed_before = 0.0
                gpu_before = []
                best_model_state = None
                start_epoch = 1

        monitor = GpuMonitor()
        monitor.start()
        started = time.time()
        for epoch in range(start_epoch, max_epochs + 1):
            model.train()
            epoch_started = time.time()
            loss_sum = 0.0
            batch_count = 0
            for batch_images, batch_labels in train_loader:
                batch_images = batch_images.to("cuda", non_blocking=True)
                batch_labels = batch_labels.to("cuda", non_blocking=True)
                mixed_images, targets = apply_mix(
                    batch_images, batch_labels, recipe["mix_mode"], rng, hp
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(mixed_images)
                    if not torch.isfinite(logits).all():
                        raise RuntimeError("non-finite training logits")
                    loss = soft_target_loss(
                        logits.float(),
                        targets,
                        gamma,
                        hp["label_smoothing"],
                        alpha_tensor,
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                loss_sum += float(loss.item())
                batch_count += 1
            validation_f1 = (
                evaluate(model, valid_loader) if valid_loader is not None else None
            )
            epoch_seconds = time.time() - epoch_started
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "loss": loss_sum / max(batch_count, 1),
                    "monitor_macro_f1": validation_f1,
                    "epoch_seconds": epoch_seconds,
                }
            )
            print(
                f"{run_key} ep{epoch:02d} loss={loss_sum / max(batch_count, 1):.4f} "
                f"monitor_f1={validation_f1} ({epoch_seconds:.0f}s)",
                flush=True,
            )
            if early_stopping:
                assert validation_f1 is not None
                if validation_f1 > best_f1:
                    best_f1 = validation_f1
                    best_epoch = epoch
                    wait = 0
                    best_model_state = _cpu_state_dict(model)
                    atomic_torch_save(
                        {
                            "model": best_model_state,
                            "model_name": model_name,
                            "contract_sha256": contract["contract_sha256"],
                            "run_key": run_key,
                            "epoch": epoch,
                            "monitor_macro_f1": validation_f1,
                            "monitor_partition": monitor_partition,
                            "outer_heldout_used_for_training_or_selection": False,
                        },
                        checkpoint,
                    )
                else:
                    wait += 1
            else:
                best_epoch = epoch
                best_f1 = float("nan")
            state = {
                "model": model.state_dict(),
                "best_model": best_model_state if early_stopping else None,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "contract_sha256": contract["contract_sha256"],
                "run_key": run_key,
                "model_name": model_name,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_f1": best_f1,
                "wait": wait,
                "epochs": epoch_rows,
                "elapsed_seconds": elapsed_before + time.time() - started,
                "gpu_samples": gpu_before + list(monitor.samples),
                "mix_rng_state": rng.bit_generator.state,
                "augment_rng_state": train_dataset.rng.bit_generator.state,
                "loader_rng_state": loader_generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                ),
            }
            atomic_torch_save(state, resume)
            if early_stopping and wait >= hp["early_stop_patience"]:
                print(f"{run_key} early stop at epoch {epoch}", flush=True)
                break

        current_gpu = monitor.finish()
        if not early_stopping:
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "model_name": model_name,
                    "contract_sha256": contract["contract_sha256"],
                    "run_key": run_key,
                    "epoch": max_epochs,
                    "fixed_epoch": max_epochs,
                    "monitor_partition": "none",
                    "metacal_or_test_labels_used": False,
                },
                checkpoint,
            )
        if not checkpoint.is_file() or best_epoch < 1:
            raise RuntimeError("训练结束但无有效 checkpoint")
        all_gpu = gpu_before + current_gpu
        record: dict[str, Any] = {
            "status": "complete",
            "contract_sha256": contract["contract_sha256"],
            "run_key": run_key,
            "stage": stage,
            "model": model_key,
            "model_name": model_name,
            "fold": fold,
            "ablation_arm": arm,
            "recipe": recipe,
            "seed": seed,
            "max_epochs": max_epochs,
            "scheduler_horizon_epochs": scheduler_horizon,
            "scheduler_rescaled_to_fixed_terminal_epoch": False,
            "early_stopping": early_stopping,
            "monitor_partition": monitor_partition,
            "outer_heldout_labels_used_for_gradient_early_stop_or_epoch_selection": False,
            "best_epoch": best_epoch,
            "best_monitor_macro_f1": (best_f1 if np.isfinite(best_f1) else None),
            "stopped_epoch": int(epoch_rows[-1]["epoch"]),
            "train_images": int(len(train_indices)),
            "monitor_images": int(len(valid_indices)),
            "class_counts_train": counts.tolist(),
            "epochs": epoch_rows,
            "total_seconds": elapsed_before + time.time() - started,
            "gpu_utilization_samples": [int(round(value)) for value in all_gpu],
            "hparams": hp,
            "environment": legacy.env_report(),
            "environment_lock": current_environment,
            "pretrained_backbone_sha256": contract["runtime_preflight"][
                "pretrained_backbone_sha256"
            ][model_key],
            "determinism": current_environment["determinism"],
            "optimizer_parameter_groups": optimizer_group_metadata,
        }
        if stage == "terminal":
            record["outer_best_epochs"] = fold_epochs
            record["terminal_epoch_rule"] = contract["epoch_selection"]["terminal"]
        atomic_write_json(log_path, record)
        resume.unlink(missing_ok=True)
        print(f"{run_key} complete", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["outer", "terminal", "ablation"], required=True
    )
    parser.add_argument("--model", choices=list(TRAIN_MODELS))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--arm", choices=["cb_only", "mixup_only", "cutmix_only"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    contract = load_contract()
    model = args.model
    if args.stage == "outer":
        if model is None or args.fold not in range(OUTER_FOLDS):
            parser.error("outer 需 --model 和 --fold 0..4")
    elif args.stage == "terminal":
        if model is None or args.fold is not None:
            parser.error("terminal 需 --model，不接受 --fold")
    else:
        if args.arm is None:
            parser.error("ablation 需 --arm")
        model = contract["additional_ablation_backbone"]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "model": model,
                    "fold": args.fold,
                    "arm": args.arm,
                    "contract_sha256": contract["contract_sha256"],
                },
                indent=2,
            )
        )
        return 0
    run_training(args.stage, str(model), args.fold, args.arm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
