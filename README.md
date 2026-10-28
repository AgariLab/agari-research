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

在同一父目录检出两个公共仓库，固定公共工具版本后进入研究仓库；
复现实验时还应检出实验所记录的研究代码 commit，检查其实现摘要与模型元数据一致。

```sh
git clone https://github.com/AgariLab/agari-kit.git
git -C agari-kit checkout c3d7b343f9c2bf85e9bda6fb3015a6a90bc69080
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
当前阶段结束默认遍历完整验证分区，按样本数加权汇总各批 loss、准确率及 CQL 损失分量；
`smoke.json` 显式设置 `validation.max_samples=128`，保持短验收有界。
日志中的 `validation_scope`、`validation_samples` 和 `validation_sample_limit` 区分完整与有界验证。
这改变了旧版 local-small 的前 128 样本验证行为，下面保留的历史数字仍对应旧实现。
默认 CPU、1 线程；MPS 必须在配置中显式选择，可用性和数值结果需在目标机器验证。

`configs/full-coverage-small.json` 保持 local-small 的模型、种子、优化器参数和 PPO 4 次更新，
把 GRP/CQL 预算改为各完整 1 个 epoch。训练前从已验证的 seeded action trace 逐场重放并计数，
跳过只有一个合法动作的强制决策。CQL 计数与实际训练使用同一读取器；每个训练半庄都消费一次；
末尾不足 batch 的样本独立更新，即使只剩 1 个样本也保留，不从下一轮补齐。
每阶段只能设置 `steps` 或 `epochs` 之一；PPO 的 `epochs` 仍表示每次采样后的优化轮数。

可先生成完整预算而不启动训练，再使用相同冻结清单运行：

```sh
python3 ../agari-kit/scripts/with_storage.py uv run --frozen agari-train plan \
  --config configs/full-coverage-small.json \
  --split datasets/manifests/selfplay-v1/split-v1.json \
  --output runs/full-coverage-trace-preflight-v1/execution-plan.json
python3 ../agari-kit/scripts/with_storage.py uv run --frozen agari-train run \
  --config configs/full-coverage-small.json \
  --split datasets/manifests/selfplay-v1/split-v1.json --run full-coverage-trace-1000-v1
```

`plan` 与训练均独立重放训练/验证数据，不打开测试牌谱。实际训练重新计算计划，检查点绑定计划摘要，
恢复时要求逐场计数、目标预算、样本游标全部一致；余弦调度器周期来自实际更新总数。
`execution-plan.json` 保存逐场原生样本数和总预算，`coverage.json` 保存实际消费样本数、完整轮次、
覆盖半庄数及逐场消费数。恢复直接跳到当前半庄，不重新计算已完成轮次的特征和奖励。

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
PPO 的恢复边界是完整更新。`checkpoint_interval_updates` 默认 1，全覆盖配置为 100，
按三阶段累计更新数计数；每个完整 epoch、阶段结束及 `--stop-after-updates` 都强制落盘。
异常退出从最近已保存的更新恢复，最多重新执行间隔内尚未保存的更新；日志可能保留重复指标。
已有实验目录不会被新训练覆盖。旧代码检查点仍受实现摘要约束，不能直接跨版本恢复。

所有运行产物位于 `AGARILAB_STORAGE_ROOT`，默认 `/Volumes/新加卷/AgariLab`：

- `runs/<run>/`：配置、冻结清单、执行计划、实际覆盖明细、JSONL 指标、TensorBoard、最终状态。
- `checkpoints/<run>/latest.pt`：按配置间隔及强制边界原子替换的完整恢复检查点。
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
当前 CQL 样本语义版本为 `seeded-trace-player-decisions-v1`，仅接受带完整 seeded trace 的本地自采数据。
原版 RiichiEnv 0.4.10 重放每一步，逐玩家检查实际 Observation 摘要与提交动作的合法性；只有一个合法动作的
决策不作为样本。所有参与者先记录选择，再执行优先级结算，因此被碰覆盖的吃和双荣的两名赢家都保留原动作。
历史 `Kyoku.steps` 会从最终 MJAI 事件推断未执行玩家的 PASS，且双荣仅产出首位赢家样本，不能把这些旧标签
等同于 collector 的真实选择。下文及既有实验目录的历史结果仍对应原读取器，未被回写为新语义验收。

CQL 按局、按玩家构造原算法的折扣回报，策略编码、网络和损失保持原定义。局边界使用玩家可见的
`round_wind/kyoku_index/honba`，逐局与原生 GRP 的起始局面和结算分数校验，包含连庄与流局。
极端 `honba=255` 因原生计数饱和无法唯一识别连续局，会明确拒绝。读取器逐条消费 trace，每次仅保留一局
的决策；原生环境及 MJAI 解析器最多保留一场牌谱。完整 native log 仅在终局读取一次，区分最后一局结算
和整场结束分配的遗留供托；GRP 继续使用原生 `take_grp_features` 的局末结果。
样本语义版本写入执行计划、检查点和导出元数据，旧检查点不能跨实现恢复；冻结数据清单及历史模型不变。
采集端原生二进制摘要保留在回执和执行计划中，不要求与训练端二进制相同，因此允许 Mac 采集后在
Linux 训练；训练端必须用固定版本完整重放并匹配 Observation、MJAI 和终分，才能接受该数据。
检查点仍绑定训练端实际二进制摘要，跨平台迁移数据不意味着允许跨运行环境直接恢复优化器状态。
当前仅在本机验证了不同采集端摘要、内容一致可接受及 Observation 不一致被拒绝；尚未实测 Mac → Linux。

验收结果记录在 `configs/baseline-plan.json`，原始动作标签修复与真实数据恢复见
[trace-action-labels-v1](experiments/trace-action-labels-v1/README.md)，修复后的完整覆盖实验见
[full-coverage-trace-1000-v1](experiments/full-coverage-trace-1000-v1/README.md)。完整实力评测、未见牌谱泛化和更大训练预算需单独实验，
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
  --split datasets/manifests/selfplay-v1/split-v1.json --models models/full-coverage-trace-1000-v1 \
  --output runs/full-coverage-trace-1000-v1/new-full-validation.json
```

该历史运行对应研究 commit `f681c69bd7a61bfa25a131223b5c3786f9bd75fa`，先于新增运行环境绑定；
新恢复逻辑另用小型实验验证，历史长评测结果不会被改写成新实现的验收结果。

旧标签实验 `full-coverage-1000-v1` 随后遍历了全部 900 场的旧迭代器输出：GRP 36,228 样本、CQL 525,721 样本，
各一个完整 epoch，PPO 仍为 4 次更新。全部 50 场验证上的 GRP 准确率为 43.3333%、loss 1.25056。
新旧 PPO 在同一 250 种子 × 四座位、efficiency/random/tsumogiri 对手池上各评测 1,000 场，
平均顺位从 2.287 降至 1.651；配对种子组差值的 95% 区间为 −0.699 至 −0.578，均无非法动作。
这支持该弱对手池上的改善，尚未通过正式晋级和强对手棋力验收。
真实小数据的跨 epoch、跨阶段中断恢复再次比较 259 个张量及全部检查点状态，精确一致。
该旧模型也完成四座位本地协议预检；当时尚未连接线上服务。逐场统计、覆盖与恢复证据保存在对应历史实验目录。

以上为旧标签历史结果。修复后 `full-coverage-trace-1000-v1` 完整消费相同 900 场的
36,228 个 GRP、525,853 个真实行为策略样本，更新数为 1,133/8,217/4；
全部 50 场验证 GRP 准确率 43.3333%、loss 1.25056，CQL loss 4.79982。
混合策略数据的吃碰杠、多家荣和和九种九牌标签另有真实样本审计。
新 PPO 本地协议预检完成四个东风局、471 次请求和三种赤五探针，零非法动作和超时。
固定弱对手池 1,000 半庄无非法动作；相对旧标签整轮模型的平均顺位差 +0.011，
配对 95% 区间 [−0.034, +0.059]，尚无明确方向性结论。
随后已通过一次真实 riichi.dev 验证：120 次动作全部 accepted，服务端返回 `passed=true`，用户确认 Bot active。
2026-10-28 06:51 UTC 核对累计 70 场完整、3 次中断，全部尝试已终止且原件保留。
最近一次在连接打开阶段触发 15 秒超时，未观察到 WebSocket 握手成功或动作请求；具体网络阶段与服务器结果未知。
同一模型和推理源码下，已用 registry v2 另登记最多 30 个新尝试；整批尚未验收，不宣称连续零故障或棋力晋级。
证据见 [真实平台验证](experiments/riichi-validation-20261028-v1/README.md) 和
[首批 100 场排位](experiments/riichi-ranked-100-20261028-v1/README.md)。

`scripts/benchmark_devices.py` 另测得本机 local-small(batch64) 的合成 CQL 更新中位耗时：
CPU 13.67ms、MPS 5.63ms。此测量仅含网络训练计算，不含解析或自博弈；尚未以完整 MPS 训练验收替代 CPU 结果。

GitHub CI 将本仓库与 `AgariLab/agari-kit` 并列检出，用锁文件安装包括 PyTorch 的完整环境后运行测试。
公共工具的 CI 引用固定为 `c3d7b343f9c2bf85e9bda6fb3015a6a90bc69080`；本机仍使用相邻工作目录。

独立进程的在线协议兼容性可以在本机预检，不需要 Token，也不会连接 riichi.dev：

```sh
python3 ../agari-kit/scripts/with_storage.py uv run --frozen python scripts/verify_online_worker.py \
  --agent models/full-coverage-trace-1000-v1/ppo-agent.json \
  --output runs/full-coverage-trace-1000-v1-online-preflight/new-report.json --targeted-red-probes
```

预检使用真实引擎 Observation、完整东风局及当前导出模型，通过 `ProcessRunner` 独立进程完成请求和动作往返；
额外从真实牌局寻找三个花色的单一合法赤五弃牌状态。回执由本机模拟，报告明确区分本地完成和服务器未验证。
