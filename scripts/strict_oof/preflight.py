"""正式训练前一次性冻结环境、预训练主干并做确定性 GPU 冒烟。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from strict_oof.protocol import (
    IMAGE_SIZE,
    MODEL_ARCHITECTURES,
    NUM_CLASSES,
    SEED,
    atomic_write_json,
    file_sha256,
    paths,
    source_code_sha256,
)

REPORT_NAME = "runtime_preflight.json"
DETERMINISM = {
    "cublas_workspace_config": ":4096:8",
    "deterministic_algorithms": True,
    "warn_only": False,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "allow_tf32": False,
    "float32_matmul_precision": "highest",
}


def configure_determinism() -> dict[str, Any]:
    """以失败关闭方式设置本协议唯一确定性策略。"""
    expected = DETERMINISM["cublas_workspace_config"]
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current != expected:
        raise RuntimeError(
            f"CUBLAS_WORKSPACE_CONFIG must be set before process start: {expected}"
        )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=False)
    observed = {
        "cublas_workspace_config": current,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if observed != DETERMINISM:
        raise RuntimeError(f"determinism configuration mismatch: {observed}")
    return observed


def model_state_fingerprint(model: torch.nn.Module) -> dict[str, Any]:
    """仅排除 ``get_classifier`` 精确模块，冻结其余预训练张量。"""
    classifier = model.get_classifier()
    prefixes = [
        name for name, module in model.named_modules() if module is classifier and name
    ]
    if len(prefixes) != 1:
        raise RuntimeError(f"classifier module prefix is not unique: {prefixes}")
    prefix = prefixes[0]
    excluded = {
        name
        for name in model.state_dict()
        if name == prefix or name.startswith(prefix + ".")
    }
    if not excluded:
        raise RuntimeError("classifier exclusion has no state keys")
    digest = hashlib.sha256()
    included = 0
    for name, tensor in sorted(model.state_dict().items()):
        if name in excluded:
            continue
        value = tensor.detach().cpu().contiguous()
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(
            value
        ).all():
            raise RuntimeError(f"non-finite pretrained tensor: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
        included += 1
    if included == 0:
        raise RuntimeError("pretrained backbone hash has no tensors")
    return {
        "sha256": digest.hexdigest(),
        "classifier_prefix": prefix,
        "included_key_count": included,
        "excluded_key_count": len(excluded),
        "excluded_keys_sha256": hashlib.sha256(
            "\n".join(sorted(excluded)).encode("utf-8")
        ).hexdigest(),
    }


def classifier_parameter_partition(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], dict[str, Any]]:
    """按 ``get_classifier`` 返回的精确模块划分骨干/分类头参数。

    不使用 ``fc``/``head`` 等子串启发式，避免将 Transformer MLP
    或 ConvNeXt ``head.norm`` 等骨干层误分到高学习率组。
    """
    classifier = model.get_classifier()
    prefixes = [
        name for name, module in model.named_modules() if module is classifier and name
    ]
    if len(prefixes) != 1:
        raise RuntimeError(f"classifier module prefix is not unique: {prefixes}")
    prefix = prefixes[0]
    classifier_ids = {id(parameter) for parameter in classifier.parameters()}
    if not classifier_ids:
        raise RuntimeError("classifier module has no parameters")
    head: list[torch.nn.Parameter] = []
    body: list[torch.nn.Parameter] = []
    head_names: list[str] = []
    body_names: list[str] = []
    seen_head_ids: set[int] = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in classifier_ids:
            if not (name == prefix or name.startswith(prefix + ".")):
                raise RuntimeError(f"classifier parameter outside exact prefix: {name}")
            head.append(parameter)
            head_names.append(name)
            seen_head_ids.add(id(parameter))
        else:
            body.append(parameter)
            body_names.append(name)
    if seen_head_ids != classifier_ids or not body:
        raise RuntimeError("classifier/body parameter partition is incomplete")
    all_ids = [id(parameter) for parameter in [*body, *head]]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("optimizer parameter partition contains duplicates")
    metadata = {
        "classifier_prefix": prefix,
        "classifier_parameter_count": len(head_names),
        "classifier_parameter_names_sha256": hashlib.sha256(
            "\n".join(sorted(head_names)).encode("utf-8")
        ).hexdigest(),
        "backbone_parameter_count": len(body_names),
        "backbone_parameter_names_sha256": hashlib.sha256(
            "\n".join(sorted(body_names)).encode("utf-8")
        ).hexdigest(),
    }
    return body, head, metadata


def model_state_sha256(model: torch.nn.Module) -> str:
    """向训练器提供冻结预训练主干哈希。"""
    return str(model_state_fingerprint(model)["sha256"])


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _command_text(args: list[str]) -> str:
    result = subprocess.run(
        args, capture_output=True, text=True, timeout=30, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"environment command failed: {args}")
    return result.stdout.strip()


def environment_lock() -> dict[str, Any]:
    """返回不含路径/用户名的可复现环境锁。"""
    freeze = _command_text([sys.executable, "-m", "pip", "freeze"])
    driver = _command_text(
        ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"]
    )
    capability = torch.cuda.get_device_capability()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: _package_version(name)
            for name in (
                "torch",
                "torchvision",
                "timm",
                "numpy",
                "scikit-learn",
                "Pillow",
                "scipy",
                "matplotlib",
            )
        },
        "cuda": torch.version.cuda,
        "cudnn": int(torch.backends.cudnn.version()),
        "driver_gpu": driver,
        "compute_capability": list(capability),
        "torch_config_sha256": hashlib.sha256(
            torch.__config__.show().encode("utf-8")
        ).hexdigest(),
        "pip_freeze_sha256": hashlib.sha256(freeze.encode("utf-8")).hexdigest(),
        "determinism": configure_determinism(),
    }


def assert_frozen_runtime(contract: dict[str, Any]) -> dict[str, Any]:
    """数值生产者启动时重验确定性设置与完整环境锁。"""
    observed = environment_lock()
    expected = contract.get("runtime_preflight", {}).get("environment_lock")
    if observed != expected:
        raise RuntimeError("runtime environment drifted from frozen preflight")
    return observed


def _all_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_runtime_preflight(
    report: Any,
    *,
    expected_source_sha256: str,
    expected_environment: dict[str, Any],
) -> list[str]:
    """深验环境、6架构真实配方冒烟和预训练指纹证据。"""
    errors: list[str] = []
    models = set(MODEL_ARCHITECTURES)
    hashes = (
        report.get("pretrained_backbone_sha256", {}) if isinstance(report, dict) else {}
    )
    fingerprints = (
        report.get("pretrained_fingerprint_metadata", {})
        if isinstance(report, dict)
        else {}
    )
    smoke = report.get("smoke", {}) if isinstance(report, dict) else {}
    if (
        not isinstance(report, dict)
        or report.get("status") != "complete"
        or report.get("all_finite") is not True
        or report.get("strict_source_code_sha256") != expected_source_sha256
        or report.get("environment_lock") != expected_environment
        or set(hashes) != models
        or set(fingerprints) != models
        or set(smoke) != models
    ):
        errors.append("runtime preflight top-level contract mismatch")
        return errors

    def smoke_row_valid(row: Any) -> bool:
        if not isinstance(row, dict):
            return False
        try:
            loss = float(row["loss"])
        except (KeyError, TypeError, ValueError):
            return False
        return bool(
            np.isfinite(loss)
            and isinstance(row.get("optimizer_parameter_count"), int)
            and row["optimizer_parameter_count"] > 0
            and row.get("optimizer_stateful_parameter_count")
            == row["optimizer_parameter_count"]
            and re.fullmatch(
                r"[0-9a-f]{64}", str(row.get("post_step_state_sha256", ""))
            )
        )

    for model_key in models:
        if re.fullmatch(r"[0-9a-f]{64}", str(hashes[model_key])) is None:
            errors.append(f"pretrained hash invalid: {model_key}")
        metadata = fingerprints[model_key]
        if not (
            isinstance(metadata, dict)
            and isinstance(metadata.get("classifier_prefix"), str)
            and int(metadata.get("included_key_count", 0)) > 0
            and int(metadata.get("excluded_key_count", 0)) > 0
            and re.fullmatch(
                r"[0-9a-f]{64}", str(metadata.get("excluded_keys_sha256", ""))
            )
            and int(metadata.get("classifier_parameter_count", 0)) > 0
            and int(metadata.get("backbone_parameter_count", 0)) > 0
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(metadata.get("classifier_parameter_names_sha256", "")),
            )
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(metadata.get("backbone_parameter_names_sha256", "")),
            )
        ):
            errors.append(f"pretrained fingerprint metadata invalid: {model_key}")
        model_smoke = smoke[model_key]
        if not isinstance(model_smoke, dict) or "strict_recipe" not in model_smoke:
            errors.append(f"strict recipe smoke missing: {model_key}")
        elif not smoke_row_valid(model_smoke["strict_recipe"]):
            errors.append(f"strict recipe smoke non-finite: {model_key}")
    representative = report.get("representative_recipe_matrix", {})
    if set(representative) != {"ce", "cb_only", "mixup_only", "cutmix_only"}:
        errors.append("representative production-loss recipe matrix incomplete")
    elif any(not smoke_row_valid(row) for row in representative.values()):
        errors.append("representative recipe matrix non-finite")
    reproducibility = report.get("same_seed_reproducibility", {})
    if not (
        reproducibility.get("initial_state_bitwise_equal") is True
        and reproducibility.get("loss_bitwise_equal") is True
        and reproducibility.get("post_step_state_bitwise_equal") is True
    ):
        errors.append("same-seed deterministic reproduction evidence failed")
    return errors


def _production_smoke_step(
    model: torch.nn.Module,
    recipe: str,
    seed: int,
    hparams: dict[str, Any],
) -> dict[str, Any]:
    """调用生产 ``apply_mix``/``soft_target_loss`` 完成一个真实优化步。"""
    from strict_oof.train import apply_mix, soft_target_loss

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    images = torch.rand(
        (2, 3, IMAGE_SIZE, IMAGE_SIZE), generator=generator, device="cuda"
    )
    labels = torch.tensor([0, 1], device="cuda")
    modes = {
        "strict_recipe": ("alternating", float(hparams["focal_gamma"]), None),
        "ce": ("none", 0.0, None),
        "cb_only": ("none", 0.0, torch.linspace(0.5, 1.5, NUM_CLASSES)),
        "mixup_only": ("mixup", 0.0, None),
        "cutmix_only": ("cutmix", 0.0, None),
    }
    mode, gamma, alpha = modes[recipe]
    local_hparams = dict(hparams)
    if mode != "none":
        local_hparams["mix_prob"] = 1.0
    rng = np.random.default_rng(seed)
    mixed, targets = apply_mix(images, labels, mode, rng, local_hparams)
    body_parameters, head_parameters, _partition = classifier_parameter_partition(model)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": body_parameters,
                "lr": float(hparams["base_lr"])
                * float(hparams["backbone_lr_mult"]),
            },
            {"params": head_parameters, "lr": float(hparams["base_lr"])},
        ],
        weight_decay=float(hparams["weight_decay"]),
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(mixed)
        loss = soft_target_loss(
            logits.float(),
            targets,
            gamma,
            float(hparams["label_smoothing"]),
            alpha,
        )
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise RuntimeError(f"non-finite production smoke: {recipe}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    optimizer_parameter_count = sum(
        len(group["params"]) for group in optimizer.param_groups
    )
    optimizer_stateful_parameter_count = len(optimizer.state)
    if optimizer_stateful_parameter_count != optimizer_parameter_count:
        raise RuntimeError(
            "production smoke AdamW state does not cover every parameter: "
            f"{optimizer_stateful_parameter_count}/{optimizer_parameter_count}"
        )
    result = {
        "loss": float(loss.detach().cpu()),
        "post_step_state_sha256": _all_state_sha256(model),
        "optimizer_parameter_count": optimizer_parameter_count,
        "optimizer_stateful_parameter_count": optimizer_stateful_parameter_count,
    }
    del optimizer
    return result


def run_preflight() -> dict[str, Any]:
    """冻结6骨干并执行生产loss分支与同种子位级复现冒烟。"""
    if not torch.cuda.is_available():
        raise RuntimeError("strict preflight requires CUDA")
    p = paths()
    target = p.out / REPORT_NAME
    import timm

    environment = environment_lock()
    initial_hashes: dict[str, str] = {}
    fingerprint_metadata: dict[str, dict[str, Any]] = {}
    smoke: dict[str, Any] = {}
    hparams = dict(__import__("config").HPARAMS)
    for model_index, (model_key, architecture) in enumerate(
        MODEL_ARCHITECTURES.items()
    ):
        seed = SEED + model_index * 1_000
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = timm.create_model(
            architecture, pretrained=True, num_classes=NUM_CLASSES
        )
        fingerprint = model_state_fingerprint(model)
        initial_hashes[model_key] = str(fingerprint.pop("sha256"))
        _body, _head, optimizer_partition = classifier_parameter_partition(model)
        fingerprint.update(optimizer_partition)
        fingerprint_metadata[model_key] = fingerprint
        model.to("cuda").train()
        smoke[model_key] = {
            "strict_recipe": _production_smoke_step(
                model, "strict_recipe", seed + 17, hparams
            )
        }
        del model
        torch.cuda.empty_cache()
    representative_key, representative_arch = next(iter(MODEL_ARCHITECTURES.items()))
    recipe_matrix: dict[str, Any] = {}
    for recipe_index, recipe in enumerate(
        ("ce", "cb_only", "mixup_only", "cutmix_only")
    ):
        seed = SEED + 90_000 + recipe_index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = (
            timm.create_model(
                representative_arch, pretrained=True, num_classes=NUM_CLASSES
            )
            .to("cuda")
            .train()
        )
        recipe_matrix[recipe] = _production_smoke_step(model, recipe, seed, hparams)
        del model
        torch.cuda.empty_cache()
    same_seed_runs = []
    reproducibility_seed = SEED + 99_999
    for _repeat in range(2):
        torch.manual_seed(reproducibility_seed)
        torch.cuda.manual_seed_all(reproducibility_seed)
        model = (
            timm.create_model(
                representative_arch, pretrained=True, num_classes=NUM_CLASSES
            )
            .to("cuda")
            .train()
        )
        initial = _all_state_sha256(model)
        result = _production_smoke_step(
            model, "strict_recipe", reproducibility_seed, hparams
        )
        same_seed_runs.append({"initial": initial, **result})
        del model
        torch.cuda.empty_cache()
    report = {
        "status": "complete",
        "strict_source_code_sha256": source_code_sha256(p.exp),
        "environment_lock": environment,
        "pretrained_backbone_sha256": initial_hashes,
        "pretrained_fingerprint_metadata": fingerprint_metadata,
        "smoke": smoke,
        "representative_recipe_model": representative_key,
        "representative_recipe_matrix": recipe_matrix,
        "same_seed_reproducibility": {
            "model": representative_key,
            "initial_state_bitwise_equal": (
                same_seed_runs[0]["initial"] == same_seed_runs[1]["initial"]
            ),
            "loss_bitwise_equal": (
                same_seed_runs[0]["loss"] == same_seed_runs[1]["loss"]
            ),
            "post_step_state_bitwise_equal": (
                same_seed_runs[0]["post_step_state_sha256"]
                == same_seed_runs[1]["post_step_state_sha256"]
            ),
        },
        "models": list(MODEL_ARCHITECTURES),
        "recipes": ["strict_recipe", "ce", "cb_only", "mixup_only", "cutmix_only"],
        "all_finite": True,
    }
    atomic_write_json(target, report)
    report["report_sha256"] = file_sha256(target)
    return report


def ensure_preflight() -> dict[str, Any]:
    """复用同源码、同环境的已完成预检，否则重新完整执行。"""
    p = paths()
    target = p.out / REPORT_NAME
    if target.is_file():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
            if not validate_runtime_preflight(
                existing,
                expected_source_sha256=source_code_sha256(p.exp),
                expected_environment=environment_lock(),
            ):
                return existing
        except (
            OSError,
            ValueError,
            RuntimeError,
            TypeError,
            AttributeError,
            json.JSONDecodeError,
        ):
            pass
    return run_preflight()
