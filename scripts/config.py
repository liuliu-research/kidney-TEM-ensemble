"""SR 大修重做实验 · 全局配置（唯一真相源）.

11 类主任务全功效集（v2.0 定版口径, 2026-06-16）。
所有脚本 import 本文件；禁止在别处硬编码类别/路径/超参。
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------- 路径（SR_WORKSPACE/SR_EXP_DIR/SR_DATA_ROOT 环境变量可覆盖，供本地演练） ----------------
REMOTE_WORKSPACE = Path(os.environ.get("SR_WORKSPACE", "/workspace"))
DATA_ROOT = Path(os.environ.get(
    "SR_DATA_ROOT", REMOTE_WORKSPACE / "0病理数据清洗与分类" / "肾病分类"))
PROJECT_DIR = (
    REMOTE_WORKSPACE
    / "2学术期刊小论文"
    / "病理数据论文"
    / "3A heterogeneous deep-learning ensemble for 13-class diagnosis of "
    "kidney disease on transmission electron microscopy（图片）"
)
EXP_DIR = Path(os.environ.get("SR_EXP_DIR",
                              PROJECT_DIR / "revision_experiments"))
BIG_DIR = EXP_DIR / "big"            # 大文件（缓存/权重），已加入 .stignore 不同步
CACHE_DIR = BIG_DIR / "jpeg384"      # JPEG 缓存
CKPT_DIR = BIG_DIR / "checkpoints"   # 模型权重
PRED_DIR = BIG_DIR / "preds"         # softmax 预测（npy，较大）
OUT_DIR = EXP_DIR / "results"        # 小体积结果（json/csv/图），正常同步
LOG_DIR = EXP_DIR / "logs"

for _d in (BIG_DIR, CACHE_DIR, CKPT_DIR, PRED_DIR, OUT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------- 类别（11 类主任务全功效集，按患者数降序） ----------------
# code -> 物理文件夹名
CLASS_FOLDERS = {
    "IgAN": "IgAN-IgA肾病",
    "IgAVN": "IgAVN-IgA血管炎相关性肾炎（紫癜性肾炎）",
    "MCD": "MCD-微小病变病",
    "MN": "MN-膜性肾病",
    "MsPGN": "MsPGN-系膜增生性肾小球肾炎（非IgA）",
    "LN": "LN-狼疮肾炎",
    "EPGN": "EPGN-毛细血管内增生性肾小球肾炎",
    "DKD": "DKD-糖尿病肾脏病",
    "TBMN": "TBMN-薄基底膜肾病与Alport综合征（IV型胶原相关肾病）",
    "FSGS": "FSGS-局灶节段性肾小球硬化",
    "MPGN": "MPGN-膜增生性肾小球肾炎（IC-MPGN）",
}
CLASSES = list(CLASS_FOLDERS)
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ---------------- 划分 ----------------
SEED = 42
SPLIT_FRACS = {"train": 0.72, "metacal": 0.08, "test": 0.20}  # 患者级分层

# ---------------- 图像 ----------------
IMG_SIZE = 384
JPEG_QUALITY = 92
IMG_EXTS = {".tif", ".tiff", ".TIF", ".TIFF", ".jpg", ".jpeg", ".png"}

# ---------------- 骨干 ----------------
BACKBONES = {
    "efficientnet_b4": "tf_efficientnet_b4.ns_jft_in1k",
    "vit_base": "vit_base_patch16_384.augreg_in21k_ft_in1k",
    "swin_base": "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
    "convnext_base": "convnext_base.fb_in22k_ft_in1k_384",
    "deit_base": "deit_base_patch16_384.fb_in1k",
}
BASELINE_BACKBONE = {"resnet50": "resnet50.a1_in1k"}

# ---------------- 训练超参（如实报告到论文） ----------------
HPARAMS = {
    "batch_size": 32,
    "batch_size_effnet": 40,
    "max_epochs": 40,
    # legacy 主链曾监控 Meta-Cal；strict_oof 改为 Train 内 inner-validation。
    "early_stop_patience": 8,
    "base_lr": 1e-4,               # 分类头；骨干 0.1x
    "backbone_lr_mult": 0.1,
    "weight_decay": 1e-4,
    "warmup_frac": 0.1,
    "focal_gamma": 1.5,
    "label_smoothing": 0.05,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,
    "mix_prob": 0.5,
    "num_workers": 0,              # RAM 缓存模式
    "amp_dtype": "bfloat16",
}

# 消融批次：统一 30 epoch 预算、早停 patience 8。
# 骨干不再写死 Swin-B，改为"主训完成后按 **Meta-Cal** macro-F1 最高者"——
#   · 科学性：消融比较的是训练配方，同一骨干内比较即可成立；用实测最强骨干
#     更有说服力（配方在最强模型上依然有效），且选择依据是 Meta-Cal 而非
#     Test，不构成选择偏倚；
#   · 工期：Swin-B 实测 1726 s/轮，ViT-B 仅 655 s/轮，6 臂 × 30 轮相差约 2.2 天。
# 由 orchestrator 在消融阶段开始前写入 results/ablation_backbone.json，
# 稿件与响应信用 {{abl_backbone}} token 渲染，不硬编码骨干名。
ABLATION_EPOCH_BUDGET = 30
ABLATION_BACKBONE_FALLBACK = "swin_base"   # 日志缺失时的兜底
ABLATIONS = {
    "ce_plain":        {"sampler": False, "focal": False, "mix": False, "cb_reweight": False},
    "ce_sampler":      {"sampler": True,  "focal": False, "mix": False, "cb_reweight": False},
    "focal_only":      {"sampler": False, "focal": True,  "mix": False, "cb_reweight": False},
    "focal_sampler":   {"sampler": True,  "focal": True,  "mix": False, "cb_reweight": False},
    "full":            {"sampler": True,  "focal": True,  "mix": True,  "cb_reweight": False},
    "full_cbreweight": {"sampler": True,  "focal": True,  "mix": True,  "cb_reweight": True},
}

# ---------------- TTA（确定性顺序；前 N 个即 N-fold TTA） ----------------
TTA_SWEEP = [1, 2, 4, 8, 16]
REFERRAL_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
# legacy 融合分析常量；strict_oof 的融合拟合使用 Train outer-OOF，
# 独立 Meta-Cal 只比较两个固定候选并拟合温度/评估转诊规则。
FUSION_OOF_FOLDS = 5

# ---------------- 其他 ----------------
GPU_IDLE_UTIL_THRESHOLD = 15      # % 以下视为空闲
GPU_IDLE_CHECK_MINUTES = 10
FOREIGN_PIDS_FILE = EXP_DIR / "foreign_train_pids.txt"  # 他人训练 PID，绝不干扰


def env_report() -> dict:
    """真实环境快照（写进论文 Methods）。"""
    import platform

    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:  # noqa: BLE001
        info["torch_err"] = str(e)
    for m in ("timm", "sklearn", "numpy", "PIL"):
        try:
            mod = __import__(m)
            info[m] = getattr(mod, "__version__", "?")
        except ImportError:
            info[m] = "MISSING"
    info["omp_threads"] = os.cpu_count()
    return info
