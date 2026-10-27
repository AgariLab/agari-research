# 原始 trace 行为标签独立审计

2026-10-28：独立原始轨迹计数已完成，重放 1,045 个完整半庄，耗时 40.94 秒。
本目录保留独立审计与修复读取器验证的小型证据；重新训练结果另行登记。

审计直接使用固定原版 RiichiEnv 0.4.10，按原始 trace 的种子重放，逐步核对
Observation 摘要和完整 `Action.to_dict()`，在引擎处理同时响应的优先级前统计玩家实际提交的动作。
只有 `len(obs.legal_actions()) > 1` 的决策进入策略样本；GRP 计数为原生 `start_kyoku` 数量乘四。
所有 MJAI 事件、终局分数及输入文件摘要均与重放结果核对。测试分区只参与清单元数据校验，未打开其内容。
计数没有使用 `MjaiReplay.steps`，也没有导入修复后的 `trace_samples`。

| 数据集 / 分区 | 半庄 | GRP 样本 | 实际策略样本 | 旧 native 策略样本 | 差值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| meld-coverage-v1 / train | 90 | 3,340 | 56,429 | 56,419 | +10 |
| meld-coverage-v1 / validation | 5 | 184 | 3,196 | 3,195 | +1 |
| selfplay-v1 / train | 900 | 36,228 | 525,853 | 525,721 | +132 |
| selfplay-v1 / validation | 50 | 1,680 | 24,603 | 24,603 | 0 |

GRP 数量均与旧报告相同。逐场检查显示，新增的 143 个策略样本恰好对应 141 个额外多家荣和玩家动作
及 2 个九种九牌动作。meld 训练分区的动作分布还存在 CHI +27、PON +7、PASS −34 的差异；
验证分区为 CHI +2、PON +1、PASS −3。完整动作类别计数见 [summary.md](summary.md)。
这些分布差异不能代替旧、新标签逐样本对齐。

此前 `baseline-1000-v1`、`full-coverage-1000-v1` 和 `meld-coverage-training-v1`
使用旧读取器训练。它们记录的样本消费数和完整 epoch 仅说明遍历了旧读取器生成的样本，
**不能视作真实行为标签完整且正确的覆盖，也不能作为读取器修复后的训练或棋力结果。**
竞争鸣牌的标签错配，以及多家荣和、九种九牌动作遗漏，会影响监督数据。

独立计数为修复读取器提供外部参考，其自身不验证逐样本 82 维动作编码、合法掩码、策略特征、
奖励/回报或 GRP 特征值。计数相等不足以通过这些验收；下面分别登记修复验证，保留历史审计的原始范围。

## 修复读取器验证

研究提交 `761b4522b5343b01413d640b58444b8a3fb7568e` 改用完整验证后的原始动作轨迹，
采样发生在同时响应的优先级裁决前。计划、检查点与导出记录
`seeded-trace-player-decisions-v1`，旧检查点不能跨语义恢复。
实现 SHA-256 为 `2f6e3618a1fa7812837b042e257d654c6e4d94f95764904002dbfef0bb9afedf`。

[reader-checks/label-regressions.json](reader-checks/label-regressions.json) 绑定两个真实训练半庄的原始文件、
Observation 与特征摘要：seed 610010 的 step 17 保留被 PON 抢先的 CHI；seed 610007 的 step 71 保留两位 RON。
已有旧样本的玩家特征逐字节一致，第二位 RON 原先没有样本。
[reader-checks/mixed-train-audit.json](reader-checks/mixed-train-audit.json) 核对全部 90 场的实际编码、
合法 mask、特征/回报有限性与计数，共 56,429 个策略样本。

29 项测试通过，包括真实动作回归、终局供托结算、单样本尾批、跨轮次样本恢复和跨采集二进制的兼容检查。
采集端二进制不同不直接阻止新训练；当前引擎必须完整重放并匹配每个 Observation、动作、全部事件与终分。
真实回归也证明：即使重新计算所有文件摘要，错误 Observation 仍被拒绝。
这不等于已实测 Mac→Linux 迁移；检查点恢复仍要求训练运行环境一致。
极端 `honba=255` 因无法安全区分连续局而明确拒绝。纯 MJAI 缺少被抢先动作意图，未作为该读取器的输入。

[native-resume/](native-resume/README.md) 使用真实 18 场训练数据，GRP/CQL 各两轮、PPO 一次更新。
两次中断恢复后，259 个张量与全部其余检查点字段（仅排除运行名）和连续运行精确一致；
两个阶段每轮的 15/30 样本尾批均保留。

[mixed-training/](mixed-training/README.md) 保存修复后的 `meld-coverage-trace-v1`：
GRP 105、CQL 882、PPO 1 次更新，完整消费 90 场的 3,340/56,429 个样本。
完整 5 场验证为 184/3,196 个样本，独立验证与训练末尾结果一致；
这是训练流程与覆盖验收，没有以该混合数据模型开展棋力比较。
原 1,000 场数据的整轮重训与固定条件对战另见
[full-coverage-trace-1000-v1](../full-coverage-trace-1000-v1/README.md)。

本次固定数据清单摘要：

- `meld-coverage-v1`：`b236cc685a0633edff33c87ae4412512689b2063ef1e5f7e146ea9f3dedddb13`。
- `selfplay-v1`：`39194c82b1f3bb8bff219425dc0ffa06ed2f0468e94f4fbd13f73d158048e4d3`。

证据文件按原始字节复制，未携带 Token、凭证、私钥或逐场原始牌谱：

- [report.json](report.json)：运行时间、Python、原版引擎二进制、kit 源码和独立脚本摘要。
- [count-delta-decomposition.json](count-delta-decomposition.json)：逐场验证后的新增样本数量分解。
- [review-check.json](review-check.json)：独立只读复核结论和未验证范围。
- [audit.py](audit.py)：本次独立审计脚本快照。
- [evidence-sha256.json](evidence-sha256.json)：五份原始证据的 SHA-256 和字节数，供复核复制完整性。

完整外置证据位于 `runs/trace-selected-actions-audit-v1/`。`report.json` 引用的两个数据集 JSON
以及逐场 JSONL 仍在该目录，未随此精简包复制；原报告的摘要保持不变。
它们包含逐场种子、选中动作分布和 trace/MJAI/验证回执哈希。

脚本使用断言执行验证，必须关闭 Python 优化模式（不可使用 `-O` 或 `PYTHONOPTIMIZE`）。
该快照包含固定外置输入与输出路径，不要直接在已有证据目录再次运行；复现须使用匹配版本环境、
相同冻结输入及旧审计报告，并准备独立存储根保存新输出。
