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
  只在独立 Meta-Cal 选择。温度只在 Meta-Cal 拟合；预设转诊规则也只在
  Meta-Cal 评估。本次阈值网格无一达到自主准确率 0.90，因此未选择运行阈值。
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

## 训练免启动的计划检查

从公开仓库根目录运行：

```bash
(cd scripts && "${SR_PYTHON:-python}" -m strict_oof.orchestrator --dry-run)
```

该命令不启动训练，但导入配置时可能在 `SR_EXP_DIR` 下创建空目录。退出码 0 只表示
计划可列出；必须同时阅读 `preflight_ready` 与 `preflight_blockers`。在严格链自己的
image-quality、runtime preflight 和 cache inventory 尚未生成前，
`preflight_ready=false` 属于预期状态，不能被写成完整复现已就绪。

## 受控前置快照与严格接续入口

`strict_oof/run_all.sh` 不是从原始影像启动的单命令端到端流程。它是
**validated legacy prerequisite snapshot 之后唯一的严格接续编排入口**。前置快照
必须已经包含患者级 manifest/split、60,509 张 JPEG 缓存及 cache report、旧主链
`analysis.json`、`core_validation.json`、`ALL.done`/`core_validation.done`、消融骨干、
旧六臂消融结果与训练摘要，以及 frozen baseline 和 I/O profiler 结果。含患者键、
路径、marker 与 checkpoint 的前置制品受限保存，不进入公开仓库。

在受控主机先用 `validate_core_results.py` 复核前置快照；确认机器报告为 PASS 后，
再从公开仓库根目录运行：

```bash
bash scripts/strict_oof/run_all.sh
```

该入口会使用 marker 和 checkpoint 有界恢复，并执行 94 步严格接续链。父目录
`scripts/run_all.sh` 是生成前置快照的 legacy GPU 工作流；它本身不能复现修订后
严格结果，也不能替代前置快照的核心验收收据。调度方式由运行环境选择，禁止并发
启动多个编排器。

## 切换门

94 步、所有 CRC/SHA-256、数据/QC/分区/概率/分析契约全绿后，
`finalize.py` 才会原子创建：

- `results/strict_oof_v1/READY.json`；
- `results/strict_oof_ACTIVE.json`。

该指针仅发布新隔离结果位置，不会覆盖旧结果。稿件构建器必须显式选择
该指针，不存在“验收未过却静默切换”的路径。
