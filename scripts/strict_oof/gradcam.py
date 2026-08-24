"""使用严格终末 ConvNeXt-B 与 Test TTA0 分量重建 Grad-CAM。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config as legacy
from strict_oof.data import IndexedRamDataset, load_rows
from strict_oof.predict import prediction_paths
from strict_oof.protocol import (
    NUM_CLASSES,
    atomic_write_json,
    file_sha256,
    load_contract,
    paths,
    source_rows,
)
from strict_oof.train import artifact_paths, exclusive_lock

MODEL_KEY = "convnext_base"


class GradCam:
    """在 ConvNeXt 最后一个 stage 上计算预测类 Grad-CAM。"""

    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        layer.register_forward_hook(self._forward_hook)
        layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output.detach()

    def _backward_hook(
        self,
        _module: torch.nn.Module,
        _grad_inputs: tuple[torch.Tensor | None, ...],
        grad_outputs: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_outputs[0].detach()

    def __call__(self, image: torch.Tensor, class_index: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        logits[0, class_index].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hook 未捕获激活或梯度")
        activations = self.activations[0]
        gradients = self.gradients[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * activations).sum(0))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.float().cpu().numpy()


def _save_panel(
    target: Path,
    raw: np.ndarray,
    cam: np.ndarray,
    true_name: str,
    pred_name: str,
    confidence: float,
) -> None:
    """原子发布一个已去标识 Grad-CAM 面板。"""
    resized = torch.nn.functional.interpolate(
        torch.as_tensor(cam)[None, None],
        size=(legacy.IMG_SIZE, legacy.IMG_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    figure, axes = plt.subplots(1, 2, figsize=(7.6, 4), dpi=160)
    axes[0].imshow(raw)
    axes[0].set_title(
        f"true={true_name}  pred={pred_name}  p={confidence:.2f}", fontsize=9
    )
    axes[1].imshow(raw)
    axes[1].imshow(resized, cmap="jet", alpha=0.4)
    axes[1].set_title("Grad-CAM (predicted class)", fontsize=9)
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".tmp.png")
    figure.savefig(temporary, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(target)


def main() -> int:
    contract = load_contract()
    from strict_oof.preflight import assert_frozen_runtime

    assert_frozen_runtime(contract)
    p = paths()
    output = p.out / "gradcam_index.json"
    checkpoint, _resume, _train_log, _train_lock = artifact_paths("terminal", MODEL_KEY)
    bundle, _shards, _meta, _predict_lock = prediction_paths(
        "terminal", MODEL_KEY, "test"
    )
    checkpoint_hash = file_sha256(checkpoint)
    bundle_hash = file_sha256(bundle)
    if output.is_file():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "complete"
                and existing.get("contract_sha256") == contract["contract_sha256"]
                and existing.get("checkpoint_sha256") == checkpoint_hash
                and existing.get("prediction_bundle_sha256") == bundle_hash
            ):
                print("strict Grad-CAM complete, skip")
                return 0
        except (OSError, json.JSONDecodeError):
            pass
    with exclusive_lock(p.big / "locks" / "strict_gradcam.lock"):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("contract_sha256") != contract["contract_sha256"]:
            raise RuntimeError("Grad-CAM checkpoint 合同不匹配")
        with np.load(bundle, allow_pickle=False) as archive:
            if str(archive["contract_sha256"].item()) != contract["contract_sha256"]:
                raise RuntimeError("Grad-CAM TTA0 预测合同不匹配")
            probabilities = archive["probs"][0].astype(np.float32)
            labels = archive["labels"].astype(np.int64)
            patients = archive["patients"].astype(str)
            global_indices = archive["global_indices"].astype(np.int64)

        _train_rows, _metacal_rows, test_rows, _hashes = source_rows()
        indexed_rows = list(enumerate(test_rows))
        images, source_labels, source_indices, source_patients, _rows = load_rows(
            indexed_rows
        )
        if not (
            np.array_equal(labels, source_labels)
            and np.array_equal(patients, source_patients)
            and np.array_equal(global_indices, source_indices)
        ):
            raise RuntimeError("Grad-CAM Test 输入与 terminal 预测顺序不一致")

        import timm

        model = timm.create_model(
            state["model_name"], pretrained=False, num_classes=NUM_CLASSES
        )
        model.load_state_dict(state["model"])
        model.eval().to("cuda")
        gradcam = GradCam(model, model.stages[-1])
        dataset = IndexedRamDataset(
            images,
            labels,
            np.arange(len(labels), dtype=np.int64),
            train=False,
        )
        predicted = probabilities.argmax(1)
        confidence = probabilities.max(1)
        picks: list[tuple[int, str]] = []
        igan = legacy.CLASS_TO_IDX["IgAN"]
        for class_index in range(NUM_CLASSES):
            correct = np.where((labels == class_index) & (predicted == class_index))[0]
            correct = correct[np.argsort(-confidence[correct])][:2]
            picks.extend((int(index), "correct") for index in correct)
            if class_index != igan:
                missed = np.where((labels == class_index) & (predicted == igan))[0]
                missed = missed[np.argsort(-confidence[missed])][:3]
                picks.extend((int(index), "missed_as_IgAN") for index in missed)

        igan_truth_classes = {
            int(labels[index]) for index, kind in picks if kind == "missed_as_IgAN"
        }
        selection_mode = "igan_attraction"
        if len(igan_truth_classes) < 2:
            selection_mode = "stratified_misclassification_fallback"
            picked_indices = {index for index, _kind in picks}
            candidates = np.where(predicted != labels)[0]
            candidates = sorted(
                (
                    int(index)
                    for index in candidates
                    if int(index) not in picked_indices
                ),
                key=lambda index: (-float(confidence[index]), index),
            )
            represented = set(igan_truth_classes)
            for index in candidates:
                true_index = int(labels[index])
                if true_index in represented:
                    continue
                picks.append((index, "misclassified_fallback"))
                represented.add(true_index)
                if len(represented) >= 2:
                    break
            if len(represented) < 2:
                raise RuntimeError(
                    "Grad-CAM has fewer than two truth classes among component errors"
                )

        records = []
        image_dir = p.out / "gradcam"
        for case_number, (index, kind) in enumerate(picks, start=1):
            tensor, _label = dataset[index]
            cam = gradcam(tensor[None].to("cuda"), int(predicted[index]))
            true_name = legacy.CLASSES[int(labels[index])]
            pred_name = legacy.CLASSES[int(predicted[index])]
            case_id = f"gc_{case_number:03d}"
            target = image_dir / f"{case_id}.png"
            _save_panel(
                target,
                images[index],
                cam,
                true_name,
                pred_name,
                float(confidence[index]),
            )
            records.append(
                {
                    "case_id": case_id,
                    "kind": kind,
                    "selection_reason": {
                        "correct": "highest-confidence correct within truth class",
                        "missed_as_IgAN": (
                            "high-confidence component error predicted as IgAN"
                        ),
                        "misclassified_fallback": (
                            "deterministic high-confidence component-error fallback"
                        ),
                    }[kind],
                    "selection_mode": selection_mode,
                    "fallback_used": selection_mode != "igan_attraction",
                    "true": true_name,
                    "pred": pred_name,
                    "conf": float(confidence[index]),
                    "input_pipeline": "jpeg384_after_annotation_strip_mask",
                    "annotation_strip_masked_px": 28,
                    "tta_component": 0,
                    "explanation_target": "predicted_class",
                    "png_sha256": file_sha256(target),
                }
            )
            print(f"saved {case_id}", flush=True)
        if len(records) < 11:
            raise RuntimeError("Grad-CAM 案例不足11，无法生成主图与补充图")
        payload = {
            "status": "complete",
            "contract_sha256": contract["contract_sha256"],
            "model": MODEL_KEY,
            "checkpoint_sha256": checkpoint_hash,
            "prediction_bundle_sha256": bundle_hash,
            "tta_component": 0,
            "input_pipeline": "jpeg384_after_annotation_strip_mask",
            "annotation_strip_masked_px": 28,
            "identifier_fields_exported": False,
            "misclassification_selection_mode": selection_mode,
            "missed_as_IgAN_truth_class_count": len(igan_truth_classes),
            "fallback_used": selection_mode != "igan_attraction",
            "cases": records,
        }
        atomic_write_json(output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
