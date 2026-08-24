"""严格 OOF 的不可变执行计划。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from strict_oof.protocol import BACKBONES, OUTER_FOLDS, TRAIN_MODELS

EXPECTED_TOTAL_STEPS = 94
EXPECTED_TRAINING_RUNS = 39


@dataclass(frozen=True)
class Step:
    """一个可独立验收和恢复的步骤。"""

    name: str
    args: tuple[str, ...]
    gpu: bool
    retries: int

    def as_dict(self) -> dict:
        """返回 JSON 可序列化形式。"""
        value = asdict(self)
        value["args"] = list(self.args)
        return value


def build_plan() -> list[Step]:
    """构造完整计划；顺序本身属于协议。"""
    steps = [
        Step(
            "image_quality_v2",
            ("strict_oof/image_quality_v2.py", "--lpips-device", "cuda"),
            True,
            3,
        ),
        # prepare 内含 6 个冻结骨干 × 3 配方的 CUDA 前向/反向确定性预飞，
        # 必须经过与正式训练相同的 GPU 空闲门和全生命周期排他锁。
        Step("prepare_contract", ("strict_oof/prepare.py",), True, 3),
    ]
    for model in TRAIN_MODELS:
        for fold in range(OUTER_FOLDS):
            steps.append(
                Step(
                    f"outer_train_{model}_fold{fold}",
                    (
                        "strict_oof/train.py",
                        "--stage",
                        "outer",
                        "--model",
                        model,
                        "--fold",
                        str(fold),
                    ),
                    True,
                    3,
                )
            )
    for model in TRAIN_MODELS:
        for fold in range(OUTER_FOLDS):
            steps.append(
                Step(
                    f"outer_pred_{model}_fold{fold}",
                    (
                        "strict_oof/predict.py",
                        "--stage",
                        "outer",
                        "--model",
                        model,
                        "--fold",
                        str(fold),
                    ),
                    True,
                    3,
                )
            )
    for model in TRAIN_MODELS:
        steps.append(
            Step(
                f"terminal_train_{model}",
                (
                    "strict_oof/train.py",
                    "--stage",
                    "terminal",
                    "--model",
                    model,
                ),
                True,
                3,
            )
        )
    for model in TRAIN_MODELS:
        for subset in ("metacal", "test"):
            steps.append(
                Step(
                    f"terminal_pred_{model}_{subset}",
                    (
                        "strict_oof/predict.py",
                        "--stage",
                        "terminal",
                        "--model",
                        model,
                        "--subset",
                        subset,
                    ),
                    True,
                    3,
                )
            )
    steps.append(
        Step(
            "strict_gradcam",
            ("strict_oof/gradcam.py",),
            True,
            3,
        )
    )
    steps.append(Step("strict_analysis", ("strict_oof/analyze.py",), False, 1))
    for arm in ("cb_only", "mixup_only", "cutmix_only"):
        steps.append(
            Step(
                f"ablation_train_{arm}",
                (
                    "strict_oof/train.py",
                    "--stage",
                    "ablation",
                    "--arm",
                    arm,
                ),
                True,
                3,
            )
        )
        for subset in ("metacal", "test"):
            steps.append(
                Step(
                    f"ablation_pred_{arm}_{subset}",
                    (
                        "strict_oof/predict.py",
                        "--stage",
                        "ablation",
                        "--arm",
                        arm,
                        "--subset",
                        subset,
                    ),
                    True,
                    3,
                )
            )
    steps.extend(
        [
            Step(
                "extended_ablation_analysis",
                ("strict_oof/analyze_ablation.py",),
                False,
                1,
            ),
            Step(
                "validate_strict_oof",
                ("strict_oof/validate.py", "--full"),
                False,
                0,
            ),
            Step(
                "finalize_strict_oof",
                ("strict_oof/finalize.py",),
                False,
                0,
            ),
        ]
    )
    if len(steps) != EXPECTED_TOTAL_STEPS:
        raise RuntimeError(
            f"严格 OOF 计划应为 {EXPECTED_TOTAL_STEPS} 步，实际 {len(steps)}"
        )
    if len({step.name for step in steps}) != len(steps):
        raise RuntimeError("严格 OOF 步骤名重复")
    training_steps = [step for step in steps if step.args[0] == "strict_oof/train.py"]
    if len(training_steps) != EXPECTED_TRAINING_RUNS:
        raise RuntimeError(
            f"完整 GPU 训练 run 应为 {EXPECTED_TRAINING_RUNS}，"
            f"实际 {len(training_steps)}"
        )
    if any("abl_" in model for model in BACKBONES):
        raise RuntimeError("主骨干名包含消融 tag")
    return steps
