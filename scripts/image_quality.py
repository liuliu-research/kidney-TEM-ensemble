"""Quantify fidelity at every stage of the actual JPEG-384 input pipeline.

The same stratified TIFF sample is evaluated at three cumulative stages:

1. RGB full-image Lanczos resize to 384 x 384;
2. resize plus blacking the bottom 28 rows;
3. resize, mask, JPEG-q92 encoding and RGB decoding (the cached uint8 input).

Each cumulative stage is compared with the native RGB source after upsampling
to native dimensions.  The incremental JPEG comparison is also reported at
384 x 384.  SSIM and PSNR use RGB arrays; LPIPS uses AlexNet and caps only its
evaluation resolution at a documented maximum side length.  The script also
checks that a newly encoded sample is byte-identical to the JPEG cache that the
training dataset actually decoded.

Usage: python image_quality.py [--per-class 30]
Output: results/image_quality.json
"""
from __future__ import annotations

import argparse
import os
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from build_cache import (
    ANNOT_STRIP_PX,
    encode_cache_jpeg,
    prepare_cache_image,
)
from config import CACHE_DIR, CLASSES, IMG_SIZE, JPEG_QUALITY, OUT_DIR, SEED

Image.MAX_IMAGE_PIXELS = None

STAGES = (
    "resize_only_vs_native",
    "resize_mask_vs_native",
    "network_input_vs_native",
    "jpeg_vs_resize_mask",
)
METRICS = ("ssim", "psnr_db", "lpips")


class PerceptualMetric(Protocol):
    """Callable interface used by the analysis and lightweight tests."""

    name: str
    device: str
    max_side: int

    def __call__(self, reference: np.ndarray, candidate: np.ndarray) -> float:
        """Return a perceptual distance for two aligned RGB uint8 arrays."""


class AlexNetLPIPS:
    """Lazy LPIPS wrapper with a bounded evaluation resolution."""

    name = "alex"

    def __init__(self, device: str = "auto", max_side: int = 512) -> None:
        try:
            import lpips
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "LPIPS is required for the formal R2.2/R5.8 analysis"
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--lpips-device cuda requested but CUDA is unavailable")
        if max_side < 64:
            raise ValueError("--lpips-max-side must be at least 64")

        self.device = device
        self.max_side = max_side
        self._torch = torch
        self._functional = torch.nn.functional
        self._model = lpips.LPIPS(net=self.name).to(device)
        self._model.eval()

    def _tensor(self, array: np.ndarray) -> Any:
        torch = self._torch
        contiguous = np.ascontiguousarray(array)
        tensor = torch.from_numpy(contiguous).to(self.device, dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        height, width = array.shape[:2]
        longest = max(height, width)
        if longest > self.max_side:
            scale = self.max_side / longest
            target = (
                max(1, round(height * scale)),
                max(1, round(width * scale)),
            )
            tensor = self._functional.interpolate(
                tensor,
                size=target,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return tensor

    def __call__(self, reference: np.ndarray, candidate: np.ndarray) -> float:
        if reference.shape != candidate.shape:
            raise ValueError(
                f"LPIPS arrays must align: {reference.shape} != {candidate.shape}"
            )
        with self._torch.inference_mode():
            value = self._model(
                self._tensor(reference), self._tensor(candidate)
            ).item()
        return float(value)


def metric_triplet(
    reference: np.ndarray,
    candidate: np.ndarray,
    perceptual: PerceptualMetric,
) -> dict[str, float]:
    """Calculate RGB SSIM, PSNR and LPIPS for aligned arrays."""
    if reference.shape != candidate.shape:
        raise ValueError(
            f"metric arrays must align: {reference.shape} != {candidate.shape}"
        )
    return {
        "ssim": float(
            ssim(reference, candidate, data_range=255, channel_axis=-1)
        ),
        "psnr_db": float(psnr(reference, candidate, data_range=255)),
        "lpips": perceptual(reference, candidate),
    }


def cache_path_for_row(row: dict[str, Any], cache_dir: Path) -> Path:
    """Derive the cache path using the same rule as ``build_cache.py``."""
    source = Path(str(row["path"]))
    return cache_dir / str(row["cls"]) / str(row["pt"]) / f"{source.stem}.jpg"


def source_path_for_row(row: dict[str, Any]) -> Path:
    """Resolve a manifest source path on the current host.

    The manifest is produced on the controlled training host.  When the same
    report is re-run from the synced Windows checkout, only the workspace
    prefix changes; no archive-relative segment is guessed or rewritten.
    """
    source = Path(str(row["path"]))
    if source.is_file():
        return source
    raw = str(row["path"]).replace("\\", "/")
    source_prefix = os.environ.get("SR_SOURCE_PREFIX", "/workspace/")
    if raw.startswith(source_prefix):
        mirror_root = Path(
            os.environ.get("SR_MIRROR_ROOT", "/workspace")
        )
        mirror = mirror_root / raw.removeprefix(source_prefix)
        if mirror.is_file():
            return mirror
    return source


def _rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def evaluate_row(
    row: dict[str, Any],
    cache_dir: Path,
    perceptual: PerceptualMetric,
) -> tuple[dict[str, dict[str, float]], dict[str, float | bool | int]]:
    """Evaluate one source image and verify its actual cached network input."""
    source_path = source_path_for_row(row)
    with Image.open(source_path) as source_image:
        native = _rgb_array(source_image)
        native_size = source_image.size
        resized_image, masked_image = prepare_cache_image(source_image)

    encoded = encode_cache_jpeg(masked_image)
    cache_path = cache_path_for_row(row, cache_dir)
    actual_cache = cache_path.read_bytes()
    byte_match = encoded == actual_cache
    if not byte_match:
        raise ValueError(
            "sample cache bytes differ from the shared production encoder"
        )

    with Image.open(io.BytesIO(actual_cache)) as decoded:
        network_image = decoded.convert("RGB")

    resized = _rgb_array(resized_image)
    masked = _rgb_array(masked_image)
    network_input = _rgb_array(network_image)

    upsample = Image.Resampling.LANCZOS
    resize_up = _rgb_array(resized_image.resize(native_size, upsample))
    mask_up = _rgb_array(masked_image.resize(native_size, upsample))
    network_up = _rgb_array(network_image.resize(native_size, upsample))

    changed_pixels = np.any(resized != masked, axis=-1)
    metrics = {
        "resize_only_vs_native": metric_triplet(
            native, resize_up, perceptual
        ),
        "resize_mask_vs_native": metric_triplet(native, mask_up, perceptual),
        "network_input_vs_native": metric_triplet(
            native, network_up, perceptual
        ),
        "jpeg_vs_resize_mask": metric_triplet(
            masked, network_input, perceptual
        ),
    }
    audit = {
        "cache_byte_match": byte_match,
        "encoded_bytes": len(encoded),
        "mask_area_fraction": ANNOT_STRIP_PX / IMG_SIZE,
        "mask_changed_pixel_fraction": float(changed_pixels.mean()),
    }
    return metrics, audit


def _empty_values() -> dict[str, dict[str, list[float]]]:
    return {
        stage: {metric: [] for metric in METRICS}
        for stage in STAGES
    }


def _summarise(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    result: dict[str, float | int | None] = {
        "n": len(values),
        "n_finite": int(finite.size),
        "n_nonfinite": len(values) - int(finite.size),
        "mean": None,
        "sd": None,
        "median": None,
        "q025": None,
        "q975": None,
    }
    if finite.size:
        result.update(
            {
                "mean": float(finite.mean()),
                "sd": float(finite.std()),
                "median": float(np.median(finite)),
                "q025": float(np.quantile(finite, 0.025)),
                "q975": float(np.quantile(finite, 0.975)),
            }
        )
    return result


def _summarise_stages(
    values: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    return {
        stage: {
            metric: _summarise(metric_values)
            for metric, metric_values in stage_values.items()
        }
        for stage, stage_values in values.items()
    }


def _extend_values(
    destination: dict[str, dict[str, list[float]]],
    source: dict[str, dict[str, float]],
) -> None:
    for stage in STAGES:
        for metric in METRICS:
            destination[stage][metric].append(source[stage][metric])


def _safe_source_id(row: dict[str, Any]) -> str:
    payload = f"{row.get('cls', '')}\t{row.get('path', '')}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def analyse(
    manifest: list[dict[str, Any]],
    per_class: int,
    seed: int,
    cache_dir: Path,
    perceptual: PerceptualMetric,
) -> dict[str, Any]:
    """Run the deterministic class-stratified stage-wise analysis."""
    if per_class < 1:
        raise ValueError("--per-class must be positive")

    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, Any]]] = {name: [] for name in CLASSES}
    for row in manifest:
        suffix = Path(str(row.get("path", ""))).suffix.lower()
        if row.get("cls") in by_class and suffix in {".tif", ".tiff"}:
            by_class[str(row["cls"])].append(row)

    selected: dict[str, list[dict[str, Any]]] = {}
    for class_name in CLASSES:
        candidates = sorted(
            by_class[class_name], key=lambda row: str(row["path"])
        )
        selected[class_name] = rng.sample(
            candidates, min(per_class, len(candidates))
        )

    digest = hashlib.sha256()
    for class_name in CLASSES:
        for row in selected[class_name]:
            digest.update(f"{class_name}\t{row['path']}\n".encode("utf-8"))

    pooled = _empty_values()
    per_class_values = {name: _empty_values() for name in CLASSES}
    mask_changed: list[float] = []
    encoded_sizes: list[float] = []
    byte_matches = 0
    errors: list[dict[str, str]] = []
    successful_by_class: Counter[str] = Counter()

    for class_name in CLASSES:
        for row in selected[class_name]:
            try:
                metrics, audit = evaluate_row(row, cache_dir, perceptual)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "class": class_name,
                        "source_id_sha256": _safe_source_id(row),
                        "error_type": type(exc).__name__,
                        "message": "sample evaluation failed; source path withheld",
                    }
                )
                continue
            _extend_values(pooled, metrics)
            _extend_values(per_class_values[class_name], metrics)
            mask_changed.append(float(audit["mask_changed_pixel_fraction"]))
            encoded_sizes.append(float(audit["encoded_bytes"]))
            byte_matches += int(bool(audit["cache_byte_match"]))
            successful_by_class[class_name] += 1

    attempted = sum(len(rows) for rows in selected.values())
    succeeded = sum(successful_by_class.values())
    if succeeded == 0:
        raise RuntimeError("no sampled image could be evaluated")

    return {
        "schema_version": 2,
        "protocol": {
            "sample_seed": seed,
            "per_class_requested": per_class,
            "source_filter": [".tif", ".tiff"],
            "color_space": "RGB",
            "resize": {
                "input": "native dimensions",
                "output": [IMG_SIZE, IMG_SIZE],
                "resampler": "Pillow Lanczos",
                "isotropic_target": True,
                "crop": False,
            },
            "footer_mask": {
                "rows": ANNOT_STRIP_PX,
                "target_height": IMG_SIZE,
                "configured_area_fraction": ANNOT_STRIP_PX / IMG_SIZE,
                "fill_rgb": [0, 0, 0],
                "purpose": "reduce footer-metadata shortcut risk",
                "limitation": (
                    "does not establish that text or measurement callouts "
                    "outside the strip are absent"
                ),
            },
            "jpeg": {
                "quality": JPEG_QUALITY,
                "encoder": "Pillow production cache encoder",
                "network_input": (
                    "actual cached JPEG decoded to RGB uint8 before "
                    "ImageNet normalization"
                ),
            },
            "cumulative_comparisons": {
                "resize_only_vs_native": (
                    "resize-only upsampled to native dimensions vs native RGB"
                ),
                "resize_mask_vs_native": (
                    "resize+mask upsampled to native dimensions vs native RGB"
                ),
                "network_input_vs_native": (
                    "actual decoded cache upsampled to native dimensions "
                    "vs native RGB"
                ),
                "jpeg_vs_resize_mask": (
                    "actual decoded cache vs lossless resize+mask at 384"
                ),
            },
            "metrics": {
                "ssim": "RGB, channel_axis=-1, data_range=255",
                "psnr_db": "RGB, data_range=255",
                "lpips": {
                    "network": perceptual.name,
                    "device": perceptual.device,
                    "max_evaluation_side": perceptual.max_side,
                    "range": "[-1, 1]",
                },
            },
        },
        "sampling": {
            "eligible_tiff_by_class": {
                name: len(by_class[name]) for name in CLASSES
            },
            "selected_by_class": {
                name: len(selected[name]) for name in CLASSES
            },
            "successful_by_class": {
                name: successful_by_class[name] for name in CLASSES
            },
            "attempted": attempted,
            "succeeded": succeeded,
            "failed": len(errors),
            "ordered_sample_sha256": digest.hexdigest(),
            "errors": errors,
        },
        "cache_byte_audit": {
            "matches": byte_matches,
            "mismatches": succeeded - byte_matches,
            "all_match": byte_matches == succeeded,
            "encoded_bytes": _summarise(encoded_sizes),
        },
        "mask_audit": {
            "configured_area_fraction": ANNOT_STRIP_PX / IMG_SIZE,
            "changed_pixel_fraction": _summarise(mask_changed),
        },
        "overall": _summarise_stages(pooled),
        "per_class": {
            name: _summarise_stages(per_class_values[name])
            for name in CLASSES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--lpips-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--lpips-max-side", type=int, default=512)
    parser.add_argument(
        "--output", type=Path, default=OUT_DIR / "image_quality.json"
    )
    args = parser.parse_args()

    manifest = json.loads((OUT_DIR / "manifest.json").read_text("utf-8"))
    perceptual = AlexNetLPIPS(args.lpips_device, args.lpips_max_side)
    output = analyse(
        manifest=manifest,
        per_class=args.per_class,
        seed=args.seed,
        cache_dir=CACHE_DIR,
        perceptual=perceptual,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(output["overall"], indent=1))
    print(
        f"sample={output['sampling']['succeeded']}/"
        f"{output['sampling']['attempted']} "
        f"errors={output['sampling']['failed']} "
        f"cache_bytes_all_match={output['cache_byte_audit']['all_match']}"
    )


if __name__ == "__main__":
    main()
