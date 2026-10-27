# agari-research

AgariLab 的本机 GRP 奖励预测 → 离线 Monte Carlo CQL → 在线 PPO 训练入口。
固定上游是 [RiichiEnv v0.4.10](https://github.com/smly/RiichiEnv/tree/479c1faeb33d082965eef8198f63261a79c0fce3/riichienv-ml)，
已用 Git 复核完整 commit `479c1faeb33d082965eef8198f63261a79c0fce3`。
模型、数据编码和损失来源及修改说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
本地 `.upstream/RiichiEnv` 是忽略提交的固定源码参考，运行训练不依赖该目录。

需要 Python 3.12，锁定 PyTorch 2.9.1、NumPy 2.4.1、TensorBoard 2.20.0 和 RiichiEnv 0.4.10。
`agari-kit==0.1.0` 当前从同级工作目录解析；下列复现步骤和 CI 均把公共工具固定到同一完整 commit。
uv 的默认索引是清华源 `https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`。
存储启动器把下载缓存、临时文件、数据和模型指向已配置外置卷，卷不可用时立即失败。

在同一父目录检出两个仓库，固定公共工具版本后进入研究仓库。研究仓库需要对应私有仓库访问权限；
复现实验时还应检出实验所记录的研究代码 commit，检查其实现摘要与模型元数据一致。

```sh
git clone https://github.com/AgariLab/agari-kit.git
git -C agari-kit checkout 9e3e866a046e40fecf94d03d315c9cc000d015b1
git clone https://github.com/AgariLab/agari-research.git
cd agari-research
```

```sh
python3 ../agari-kit/scripts/with_storage.py uv sync --frozen
.venv/bin/python -m unittest discover -s tests -v
```

训练必须显式指定公共工具生成的冻结数据清单，不能把任意 MJAI 目录作为训练集。
清单按完整半庄时间顺序分成 90/5/5；训练检查清单摘要、验证回执摘要、MJAI 原始文件摘要、
规范化事件摘要及动作轨迹摘要。训练和验证只读取各自分区，测试牌谱不进入训练读取器。
读取器还会独立重放训练/验证分区的完整动作轨迹，防止人为伪造一组摘要自洽的回执。

```sh
python3 ../agari-kit/scripts/with_storage.py uv run --frozen agari-train run \
  --config configs/smoke.json \
  --split datasets/manifests/smoke-v1/split-v1.json \
  --run baseline-smoke
```

`configs/smoke.json` 使用 25,592 参数的小型 Q 网络，运行 GRP 4 次、CQL 8 次、PPO 1 次更新，
用于确认完整训练链路。`configs/local-small.json` 增加网络容量，运行 100/200/4 次更新，
每次 PPO 更新采集一个完整半庄。两份配置均为固定更新预算，不保证遍历大数据集；实际离线样本数
记录在检查点 `samples_seen`，PPO 采样场次、座位和种子写入指标。小规模训练通过不代表达到强基线水平。
阶段结束的验证指标使用验证分区固定顺序的前 128 个样本，日志同时记录实际样本数；这不是全量验证报告。
默认 CPU、1 线程；MPS 必须在配置中显式选择，可用性和数值结果需在目标机器验证。

可在任意更新后保存并停止，再使用完全相同的代码、配置和数据恢复：

```sh
python3 ../agari-kit/scripts/with_storage.py uv run --frozen agari-train run \
  --config configs/smoke.json --split datasets/manifests/smoke-v1/split-v1.json \
  --run resume-demo --stop-after-updates 6
python3 ../agari-kit/scripts/with_storage.py uv run --frozen agari-train run \
  --config configs/smoke.json --split datasets/manifests/smoke-v1/split-v1.json \
  --run resume-demo --resume checkpoints/resume-demo/latest.pt
```

检查点包含 GRP、Q、Actor-Critic、冻结对手、优化器、调度器、随机数状态、数据游标、PPO 场次计数、
配置及代码/数据/上游摘要。当前版本还把实际 `agari-kit` 源码摘要、引擎动态库摘要、Python/Torch/NumPy 版本
及 `uv.lock` 摘要写入检查点与导出元数据，恢复时逐项要求一致。缺少运行环境记录的历史检查点不能直接恢复；
历史导出模型仍可用于推理。训练需要含 `uv.lock` 的源码 checkout。CPU 使用确定性运算。
PPO 的恢复边界是完整更新；进程异常时从最近完成的更新继续，
日志可能保留检查点写入前的重复指标。已有实验目录不会被新训练覆盖。

所有运行产物位于 `AGARILAB_STORAGE_ROOT`，默认 `/Volumes/新加卷/AgariLab`：

- `runs/<run>/`：配置、冻结清单、JSONL 指标、TensorBoard、最终状态。
- `checkpoints/<run>/latest.pt`：每次更新原子替换的完整恢复检查点。
- `models/<run>/{grp,cql,ppo}.pt`：阶段完成后的权重及对应 `.json` 摘要。
- `models/<run>/{cql,ppo}-agent.json`：公共工具可加载的工厂描述，绑定权重文件 SHA-256。

```sh
python3 ../agari-kit/scripts/with_storage.py uv run --frozen agari evaluate \
  --candidate @/Volumes/新加卷/AgariLab/models/baseline-smoke/ppo-agent.json \
  --run baseline-smoke-evaluation
```

算法保留上游的 GRP 20 维结果特征、74×34 的玩家可见策略特征、82 个原生动作编码与合法掩码。
CQL 使用 GRP 预测的局末奖励形成 Monte Carlo 回报，加合法动作 `logsumexp(Q)-Q(data)` 保守项和四分类名次辅助损失。
PPO 从 Q 网络迁移骨干与决策头，以冻结模型作为对手和 KL 参考；使用裁剪目标、GAE、价值损失和熵项。
结果信息仅用于监督标签与训练奖励。`ModelAgent` 只接收玩家 `Observation`，无全局环境对象。
完整局边界由 `end_kyoku` 识别，包含连庄；GRP 特征由原生重放解析，避免手工结算重复和供托错误。

当前验收结果记录在 `configs/baseline-plan.json`。完整实力评测、未见牌谱泛化和更大训练预算需单独实验，
不能从 smoke loss 或同源规则 AI 自博弈数据推出。

2026-10-28 已使用 20 场完整引擎验证牌谱跑完两种 CPU 配置。冒烟运行
`baseline-smoke-v2` 在 CQL 第 2 次更新中断恢复，259 个检查点张量及其余全部状态精确一致；
`baseline-local-v1` 在 PPO 第 2 次更新中断恢复，385 个张量及其余状态精确一致。
两者的恢复证据分别位于外置 `runs/baseline-resume-v2/` 和 `runs/baseline-local-resume-v1/`。
`scripts/verify_resume.py` 可以重复比较全部模型、优化器、调度器、随机数与数据游标。
新增运行环境绑定后的 `baseline-runtime-smoke-v2` / `baseline-runtime-resume-v2` 同样完成 259 个张量
及全部状态精确比较，使用公共工具 commit `28894987cc121775fdbb12b22f0db72aca9b3695`。
真实旧版公共工具检查点的恢复也因源码摘要不同被拒绝，证据为后者目录下的 `old-kit-rejection.json`。
后来针对独立对手的修正不改写这次验收的运行环境记录；恢复该历史实验仍需匹配记录的环境。

20 场数据只有 1 个验证半庄。local-small 的 GRP 验证准确率为 0，不能把训练完成当作奖励模型质量合格，
也不能用这批小模型声称棋力提升。需要更大、来源合适的数据和独立对局评估。

同一训练实现随后对 `selfplay-v1` 的 1000 场数据（900/50/50 分区）运行 `baseline-1000-v1`，
仍使用 local-small 的固定更新预算。GRP 实际消费 3200 样本、涉及 80/900 场；CQL 消费 12800 样本、
涉及 23/900 场；PPO 采集 4 场、649 次玩家决策。阶段验证各取前 128 样本，仅分别涉及 4/50 与 1/50 场：
GRP 准确率 36.71875%、loss 1.37826，CQL loss 8.53376。这些数字不表示跑完一个 epoch、全量验证或棋力提升。
消费顺序与场次清单保存在外置 `runs/baseline-1000-v1/sample-coverage.json`。
随后用独立脚本完整评估 50 场验证分区：GRP 共 1680 样本，准确率 **27.0238%**、loss 1.38431；
CQL 共 24603 样本，loss 8.63535（TD 0.319209、保守项 1.613290、辅助名次项 1.248475）。
完整报告 `runs/baseline-1000-v1/full-validation.json` 绑定数据、权重和实际评估环境摘要，
未打开训练或测试牌谱内容。这些全量验证指标取代前 128 样本作为验证集质量描述，但仍不是棋力结论。

```sh
python3 ../agari-kit/scripts/with_storage.py uv run --frozen python scripts/validate_models.py \
  --split datasets/manifests/selfplay-v1/split-v1.json --models models/baseline-1000-v1 \
  --output runs/baseline-1000-v1/new-full-validation.json
```

该历史运行对应研究 commit `f681c69bd7a61bfa25a131223b5c3786f9bd75fa`，先于新增运行环境绑定；
新恢复逻辑另用小型实验验证，历史长评测结果不会被改写成新实现的验收结果。

`scripts/benchmark_devices.py` 另测得本机 local-small(batch64) 的合成 CQL 更新中位耗时：
CPU 13.67ms、MPS 5.63ms。此测量仅含网络训练计算，不含解析或自博弈；尚未以完整 MPS 训练验收替代 CPU 结果。

GitHub CI 将本仓库与 `AgariLab/agari-kit` 并列检出，用锁文件安装包括 PyTorch 的完整环境后运行测试。
公共工具的 CI 引用固定为 `9e3e866a046e40fecf94d03d315c9cc000d015b1`；本机仍使用相邻工作目录。

独立进程的在线协议兼容性可以在本机预检，不需要 Token，也不会连接 riichi.dev：

```sh
python3 ../agari-kit/scripts/with_storage.py uv run --frozen python scripts/verify_online_worker.py \
  --agent models/baseline-local-v1/ppo-agent.json \
  --output runs/baseline-local-v1-online-preflight/new-report.json --targeted-red-probes
```

预检使用真实引擎 Observation、完整东风局及当前导出模型，通过 `ProcessRunner` 独立进程完成请求和动作往返；
额外从真实牌局寻找三个花色的单一合法赤五弃牌状态。回执由本机模拟，报告明确区分本地完成和服务器未验证。
