# 原生 trace 训练断点恢复验收

使用修复后的 `seeded-trace-player-decisions-v1` 读取器，在真实 `smoke-v1` 冻结数据上运行：18 个训练半庄、1 个验证半庄；未打开测试牌谱内容。
GRP/CQL 各完成 2 个 epoch，随后完成 1 次 PPO 更新。连续运行与分两次恢复的运行仅排除 `run_name`，其余全部检查点字段精确相等，包括 259 个模型/优化器/调度器/RNG 张量及数据游标。

- 第一次在 GRP 第 36 次更新停止，位于第二个 epoch 内；已消费 1,100 个 GRP 样本。
- 第二次跨过 GRP 完成点，在 CQL 第 153 次更新停止，恰好完成第一轮尾批。
- 最终更新数为 GRP 48 / CQL 306 / PPO 1，GRP/CQL 分别消费 1,456 / 20,428 样本。
- 每轮 GRP 尾批为 15（batch 31），CQL 尾批为 30（batch 67）；两轮全部保留。验证均遍历完整验证分区。
- Cosine 调度器最终 `T_max/last_epoch` 分别为 GRP `48/48`、CQL `306/306`。

实现摘要：`2f6e3618a1fa7812837b042e257d654c6e4d94f95764904002dbfef0bb9afedf`。实际运行环境、代码 commit、配置/脚本/检查点摘要见 `resume-equivalence.json`。
完整模型、检查点和日志保存在外置 `runs/trace-native-resume-v1`、`models/trace-native-{continuous,interrupted}-v1` 和 `checkpoints/trace-native-{continuous,interrupted}-v1`；此处仅保存小型证据，没有覆盖历史 `epoch-native-*` 运行。
