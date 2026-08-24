# R4.1 strict backbone-level OOF workflow

该流程为 Scientific Reports 大修 R4.1 与配套消融意见的隔离补训链。
它不覆盖旧 `results/analysis.json`、`big/checkpoints/` 或旧 marker。

## 冻结协议

- 原 Train 内做 patient-grouped outer 5-fold。
- 每个 outer-train 内再做 patient-grouped inner validation，仅用它早停；
  outer-heldout 标签不进入梯度、早停或 epoch 选择。
- 5 个主骨干 + ResNet50 都产生 Train OOF TTA8。ResNet50 只是
  same-protocol standalone comparator，不参与 5 骨干 ensemble。
- 每个模型用 5 个 outer best epoch 的中位数，在完整 Train 上固定
  epoch 重训，不早停；scheduler 仍按 40 epoch 周期计算。
- final backbone 产生 Meta-Cal/Test TTA16。投票权重与 LR meta-learner
  只在完整 Train OOF 拟合；预注册的 voting_tta8 vs stack_lr_tta8
  只在独立 Meta-Cal 选择。温度与转诊阈值也只在 Meta-Cal 拟合。
  Test 标签只用于终局评估。
- 新增 CB-only、MixUp-only、CutMix-only 三个单独消融臂，与旧 6 臂统一为
  9 臂 schema。

## 计划与可复现性

计划包含 94 个可独立验收步骤和严格固定的 39 个完整 GPU 训练 run：

- 6 个模型 × 5 个 Train outer folds = 30；
- 6 个模型在完整 Train 上固定轮次重训 = 6；
- CB-only、MixUp-only、CutMix-only 三个附加消融臂 = 3。

运行期 checkpoint、marker 与逐 run 日志仅用于受控主机上的恢复和验收，不属于
公开包。公开包仅提供代码、经隐私门控的聚合结果和出版图。

## 只读预检（不启动训练）

```bash
python -m strict_oof.orchestrator --dry-run
```

## 唯一主复现入口

在公开仓库根目录运行：

```bash
bash scripts/strict_oof/run_all.sh
```

这是修订后分析的 sole main reproduction entry。父目录中的 legacy
`scripts/run_all.sh` 及其旧训练、推理与融合模块已被 superseded，仅为来源追溯
和兼容性保留，不得用于复现修订后结果。严格入口会使用 marker 和 checkpoint
进行有界恢复；调度方式由运行环境选择，不能同时启动多个编排器。

## 切换门

94 步、所有 CRC/SHA-256、数据/QC/分区/概率/分析契约全绿后，
`finalize.py` 才会原子创建：

- `results/strict_oof_v1/READY.json`；
- `results/strict_oof_ACTIVE.json`。

该指针仅发布新隔离结果位置，不会覆盖旧结果。稿件构建器必须显式选择
该指针，不存在“验收未过却静默切换”的路径。
