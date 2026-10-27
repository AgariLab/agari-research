# 修正 trace 标签后的 1000 场训练验收

`full-coverage-trace-1000-v1` 已完成，research 为 `761b452`，样本语义为
`seeded-trace-player-decisions-v1`。配置与 `configs/full-coverage-small.json` 及原
`full-coverage-1000-v1` 完全相同；原运行和模型未覆盖。

| 阶段 | 训练样本 | 更新次数 | 尾批样本 | 完整验证样本 |
| --- | ---: | ---: | ---: | ---: |
| GRP | 36,228 | 1,133 | 4 | 1,680 |
| CQL | 525,853 | 8,217 | 29 | 24,603 |

900 场训练各消费一次。计划及实际覆盖的每场 ID、内容摘要和样本数均与
`runs/trace-selected-actions-audit-v1/selfplay-v1-train-games.jsonl` 的独立审计一致。
PPO 完成 4 次更新；具体新增对局与决策数量见验收摘要。

独立重跑全部 50 场 validation：GRP loss `1.2505604914`、accuracy `0.4333333333`；
CQL loss `4.7998232616`。样本数及 loss 与训练阶段末尾验证完全一致。
检查点、全部导出模型和 descriptor 的语义版本、运行环境、源码、配置、权重摘要已核对。
50 场 test 内容未读取。本报告验证训练流程与产物，不作为实力评测结论。

运行目录创建时间至 `result.json` 写入时间约 **207.44 秒**，这是文件时间戳估计，包含样本计数、
训练和阶段验证，不含目录创建前的数据构造与完整原生重放，不能视为完整 CLI 耗时。
训练后的独立 validation 用时约 **7.12 秒**。

[验收摘要](acceptance.json) 记录核验范围与时间依据；[逐场覆盖](coverage.json)、
[执行计划](execution-plan.json)、[配置](config.json)、[训练结果](result.json) 和
[完整验证](validation.json) 为外置产物的原字节副本。
[产物摘要](artifacts.json) 绑定外置模型、检查点、日志和独立审计文件的 SHA-256。
模型与完整日志仍保留在外置存储。
