# -*- coding: utf-8 -*-
"""公开前隐私审计（伦理门控）。任一 BLOCK 项命中即不得公开。

移植自项目 2（report-derived TEM morphometrics）的同名脚本，按本项目
「图像 + 患者档案」的风险面扩写。与项目 2 的差异集中在下面 5 类新增阻塞规则，
因为本项目的原始资料是电镜**图像**与**患者活检文件夹**，而非文本报告聚合表：

* 患者准标识符：`PT######` 档案编号（= 患者假名键），以及 `PT######_YYYYMMDD`
  形式的活检文件夹名（假名键 + 个体级日期，两项合一构成强准标识符）
* 患者级清单文件：`split.json` / `manifest.json`（逐图逐患者的行级清单）
* 模型权重与预测张量：`*.pt/*.pth/*.ckpt/*.safetensors/*.npy/*.npz/*.pkl` 等
* 原始医学图像：`*.tif/*.tiff/*.bmp/*.dcm/*.svs/*.ndpi`（电镜原图一律不得随仓库分发）
* 超大 JSON（> 2 MB）：结构性兜底——即使文件被改名，行级清单也过不了体积这关

沿用项目 2 的规则：身份证 / 手机号 / IPv4 / 凭据 / 机构名（带必要披露豁免）/
患者姓名字段 / 绝对路径 / 行级 CSV 启发式 / 精确日期（告警不阻塞）。
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

BLOCK_PATTERNS = {
    "身份证号": re.compile(r"\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b"),
    "手机号": re.compile(r"\b1[3-9]\d{9}\b"),
    "IPv4": re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    "凭据": re.compile(r"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|private[_-]?key|webhook)\b\s*[=:]\s*\S+"),
    "机构名": re.compile(r"(?i)\b(?:xiangya|changsha|hunan)\b|湘雅|长沙|中南大学"),
    # 只认「患者姓名」类字段；不能写成裸 name，否则 .zenodo.json 的作者 name 会误报
    "患者姓名字段值": re.compile(r"(?i)\"(?:patient_name|pat_name|patientname|姓名|患者姓名)\"\s*:\s*\"[^\"]+\""),
    "绝对路径": re.compile(r"[A-Za-z]:[\\/](?:Users|Windows|Program)|/(?:home|Users)/[A-Za-z]"),
    # ↓ 本项目新增：患者假名键与活检夹名
    "患者档案编号": re.compile(r"\bPT\d{4,}\b"),
    "活检文件夹名": re.compile(r"\bPT\d+_(?:19|20)\d{6}\b"),
}

# 禁止随仓库分发的文件类型（患者级数据 / 图像 / 权重 / 预测张量）
BLOCK_SUFFIX = {
    ".tif": "原始医学图像", ".tiff": "原始医学图像", ".bmp": "原始医学图像",
    ".dcm": "原始医学图像", ".svs": "全片扫描图像", ".ndpi": "全片扫描图像",
    ".pt": "模型权重", ".pth": "模型权重", ".ckpt": "模型权重",
    ".safetensors": "模型权重", ".onnx": "模型权重", ".h5": "模型权重",
    ".npy": "预测/特征张量", ".npz": "预测/特征张量",
    ".pkl": "序列化对象", ".pickle": "序列化对象", ".joblib": "序列化对象",
    ".gpickle": "序列化对象",
}
# 位图只允许出现在 figures/（出版图），且一律转人工确认
RASTER_SUFFIX = {".png", ".jpg", ".jpeg"}

# 患者级清单文件（大小写敏感：仓库自身的 MANIFEST.json 不在此列）
BLOCK_NAMES = {"split.json", "manifest.json", "image_report_pairs.json",
               "cache_report.json", "split_stats.json"}
BIG_JSON_BYTES = 2 * 1024 * 1024

# 允许出现的日期（数据冻结、划分与运行时间戳），其余精确日期需人工确认
DATE_ALLOW = {"2026-06-16", "2026-08-08", "2026-08-09"}
DATE_RX = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")

TEXT_SUFFIX = {".py", ".md", ".csv", ".json", ".txt", ".yml", ".yaml", ".sh",
               ".cff"}
# 无扩展名但确为文本，需照常扫描（否则只会被记成「未知类型」草草放过）
TEXT_NAMES = {".gitignore", ".gitattributes", "LICENSE"}
ROW_LEVEL_HINTS = {"patient", "person", "biopsy", "visit", "admission",
                   "record_id", "stem", "filename", "image_path"}


def main() -> int:
    findings: list[str] = []
    warnings: list[str] = []
    files = [p for p in ROOT.rglob("*")
             if p.is_file() and p.name != Path(__file__).name
             and ".git" not in p.parts]

    for p in files:
        rel = p.relative_to(ROOT)
        suffix = p.suffix.lower()

        # ---- 规则 A：按文件类型阻塞（患者级数据 / 图像 / 权重）----
        if suffix in BLOCK_SUFFIX:
            findings.append(f"{rel} [{BLOCK_SUFFIX[suffix]}] 禁止随仓库分发")
            continue
        if suffix in RASTER_SUFFIX:
            if rel.parts[0] != "figures":
                findings.append(f"{rel} [位图] 仅允许出现在 figures/ 下的出版图")
            else:
                warnings.append(
                    f"{rel} [出版图需人工确认] Grad-CAM 等叠加图须确认未含"
                    f"院内标注、病例编号或可识别内容")
            continue

        # ---- 规则 B：按文件名阻塞（患者级清单）----
        if p.name in BLOCK_NAMES:
            findings.append(f"{rel} [患者级清单文件] 含 PT 编号等准标识符，禁止公开")
            continue

        # ---- 规则 C：超大 JSON 结构性兜底（改名也拦得住）----
        if suffix == ".json" and p.stat().st_size > BIG_JSON_BYTES:
            findings.append(
                f"{rel} [疑似行级/图像级清单] {p.stat().st_size} 字节 "
                f"> {BIG_JSON_BYTES}，聚合结果不应有此体量")
            continue

        if suffix not in TEXT_SUFFIX and p.name not in TEXT_NAMES:
            warnings.append(f"{rel} [未扫描的二进制/未知类型] 需人工确认")
            continue

        text = p.read_text(encoding="utf-8", errors="replace")

        # ---- 规则 D：正则内容扫描 ----
        for label, rx in BLOCK_PATTERNS.items():
            for m in rx.finditer(text):
                line = text[: m.start()].count("\n") + 1
                # 机构（湘雅医院）在论文中本就公开披露：作者单位与伦理批件均须具名。
                # 因此机构名只有出现在数据/代码文件里才构成泄漏；下面两类属必需披露：
                #   1) README 等说明文档中的伦理批件与单位声明
                #   2) 引用元数据（.zenodo.json / CITATION.cff）中的作者 affiliation
                if label == "机构名":
                    ctx = text[max(0, m.start() - 260): m.end() + 120]
                    if suffix == ".md" and re.search(
                        r"(?i)ethic|approval|affiliat|committee|consent", ctx
                    ):
                        continue
                    if p.name in (".zenodo.json", "CITATION.cff") and re.search(
                        r"(?i)affiliation", ctx
                    ):
                        continue
                findings.append(f"{rel}:{line} [{label}] {m.group(0)[:60]!r}")

        # ---- 规则 E：精确日期（告警）----
        for m in DATE_RX.finditer(text):
            if m.group(0) not in DATE_ALLOW:
                line = text[: m.start()].count("\n") + 1
                warnings.append(f"{rel}:{line} [精确日期] {m.group(0)}")

        # ---- 规则 F：行级 CSV 启发式 ----
        if suffix == ".csv":
            with p.open(encoding="utf-8-sig", newline="") as fh:
                rows = list(csv.reader(fh))
            if rows:
                header = [h.strip().lower() for h in rows[0]]
                risky = [h for h in header if any(k in h for k in ROW_LEVEL_HINTS)]
                if len(rows) - 1 > 200 and risky:
                    findings.append(f"{rel} [疑似行级数据] {len(rows)-1} 行，列 {risky}")

    print(f"扫描 {len(files)} 个文件")
    print()
    if findings:
        print("===== 阻塞项（不得公开）=====")
        for f in findings:
            print("  " + f)
    else:
        print("阻塞项：无")
    print()
    if warnings:
        print(f"===== 需人工确认（{len(warnings)} 处）=====")
        for w in warnings[:15]:
            print("  " + w)
        if len(warnings) > 15:
            print(f"  ……另有 {len(warnings)-15} 处")
    else:
        print("需人工确认：无")
    print()
    print("审计结论：" + ("不通过，禁止公开" if findings else "通过，可以公开"))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
