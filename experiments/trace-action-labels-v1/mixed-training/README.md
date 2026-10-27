# 修正 trace 标签后的混合数据训练验收

`meld-coverage-trace-v1` 已完成，使用 research `761b452` 和 `seeded-trace-player-decisions-v1`。
数据仍为原混合 100 场：训练 90 场、验证 5 场、test 5 场；test 内容未读取。配置与原
`runs/meld-coverage-audit-v1/training-config.json` 相同，历史训练产物未覆盖。

| 阶段 | 训练样本 | 更新次数 | 最后批次 | 完整验证样本 |
| --- | ---: | ---: | ---: | ---: |
| GRP | 3,340 | 105 | 12 | 184 |
| CQL | 56,429 | 882 | 45 | 3,196 |

GRP 和 CQL 均完成一整个 epoch，每个训练半庄恰好消费一次。CQL 标签来自实际提交动作，
保留被优先级覆盖的鸣牌和双荣双方选择；计数与采样使用同一 reader。
PPO 完成一次更新，新增自博弈 1 场、8 局、151 个英雄决策样本。
训练 CLI 总耗时 17.699 秒（包含入库验证、计数及阶段验证）。

独立完整验证得到 GRP loss `1.2684735837`、accuracy `0.2989130435`；CQL loss
`9.6804019107`。样本数量及 loss 与训练末尾验证一致。检查点和全部模型导出的语义版本、
运行环境、实现摘要及权重哈希已核对。这是读取器修正后的小规模训练验收，不能作为实力结论；
未进行线上验证或实际 Mac → Linux 迁移。

[验收摘要](report.json) 含命令与核验项；[覆盖记录](coverage.json) 保存逐场消费计数；
[执行计划](execution-plan.json)、[配置](config.json)、[训练结果](result.json) 和
[完整验证报告](validation.json) 为外置产物的原字节副本。
[产物摘要](artifacts.json) 记录外置检查点、模型、原始 metrics 和相关清单的 SHA-256。
此目录仅保存小型证据，模型及完整日志位于外置存储。
