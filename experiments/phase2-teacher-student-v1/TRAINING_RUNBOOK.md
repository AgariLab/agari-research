# Phase 2：本机训练与开发评测执行说明

本说明对应已登记的三臂、三个训练种子，以及随后限定范围的开发评测。下列命令是使用规程，不表示训练或评测已经启动或完成。教师资格和 1,000 局教师数据通过门禁后，准备器才会生成九个配置；配置尚未生成时，不能预先填写实际更新数，也不能把目录存在当成准备完成。本文不启动在线 Bot，不自动扩量、最终评测或 Akochan 对局。

## 固定路径与运行身份

| 项目 | 固定值 |
| --- | --- |
| 外置根目录 | `/Volumes/新加卷/AgariLab` |
| Python | `/Users/zhou/MyCodes/AgariLab/agari-research/.venv/bin/python` |
| 运行时代码根目录 | `/Users/zhou/MyCodes/AgariLab/phase2-runtime-20261029-v1` |
| 原工具仓库 | `/Users/zhou/MyCodes/AgariLab/phase2-tools-20261029-v1/agari-research` |
| 新批处理工具仓库 | `/Users/zhou/MyCodes/AgariLab/phase2-batch-tools-20261029-v1/agari-research` |
| 新批处理工具冻结 commit | `9dd2902e23429c6364071291a5556d35b312fb1f` |
| 教师资格 | `runs/phase2-teacher-qualification-v1` |
| 教师原始数据 | `datasets/manifests/phase2-teacher-1000-v1/split-v1.json` |
| 训练准备目录 | `runs/phase2-pilot-1000-v1` |
| 上游启动记录 | `runs/phase2-pilot-1000-launch-v1` |
| 训练批处理目录 | `runs/phase2-training-1000-launch-v1` |
| 开发批处理目录 | `runs/phase2-development-1000-launch-v1` |
| 小规模开发矩阵 | `runs/phase2-development-small-v1` |

原 tools 固定在 `b9ee0715c8e1c16ea811e3e37f2a8ebd89814671`，提供 pilot、单任务 runner 和 matrix；其路径和内容保持不变。新 batch tools 已另建干净的 detached worktree，固定在表中的 `9dd2902e23429c6364071291a5556d35b312fb1f`。批处理脚本来自新目录，单任务 runner/matrix 来自原 tools，包导入始终来自固定 runtime；不要把可变主仓库当作入口。两种 tools 仓库的 `src` 都不参与本轮运行。

运行时代码已固定：kit `78719f35ea17d68e3a6fcfd04a2aa969a95142cc`，research `80d3c4204d381a93a7f53d729fa59c6074ff1b42`。research implementation SHA 为 `411478677061a24abdb1ea56f185c9ab05015b841f027e30b201fec2c75c905f`。Python 3.12.16、Torch 2.9.1、NumPy 2.4.1、RiichiEnv 0.4.10；runner 同时核验实际源码、锁文件和引擎二进制身份。不要在训练或恢复间更新环境、运行时代码、配置或工具脚本。

```sh
PHASE2_PYTHON=/Users/zhou/MyCodes/AgariLab/agari-research/.venv/bin/python
PINNED_ROOT=/Users/zhou/MyCodes/AgariLab/phase2-runtime-20261029-v1
PHASE2_TOOLS=/Users/zhou/MyCodes/AgariLab/phase2-tools-20261029-v1/agari-research
PHASE2_BATCH_TOOLS=/Users/zhou/MyCodes/AgariLab/phase2-batch-tools-20261029-v1/agari-research
PHASE2_STORAGE=/Volumes/新加卷/AgariLab
PHASE2_PREPARATION=runs/phase2-pilot-1000-v1
export AGARILAB_STORAGE_ROOT="$PHASE2_STORAGE"
export PYTHONPATH="$PINNED_ROOT/agari-kit/src:$PINNED_ROOT/agari-research/src"
export PYTHONDONTWRITEBYTECODE=1
```

首批九任务统一用 CPU、`torch_threads=1`。既有 smoke 已证明 CPU 与 MPS 各自严格恢复一致；并行教师资格期间测得的短跑没有明确端到端 MPS 优势，因此不在三臂或三个种子间混用设备。这是本轮固定选择，不代表空闲机器的性能排序。

## 准备配置，只做一次

父流程使用 `phase2-pilot-1000-launch-v1` 等待资格、采集和数据门禁。下面是同一准备步骤的准确底层命令；已有完整准备目录时直接进入 inspection，不重复创建。

```sh
"$PHASE2_PYTHON" -B -u "$PINNED_ROOT/agari-research/scripts/prepare_teacher_experiment.py" \
  --weak-split datasets/manifests/selfplay-v1/split-v1.json \
  --teacher-split datasets/manifests/phase2-teacher-1000-v1/split-v1.json \
  --teacher-descriptor "$PHASE2_STORAGE/runs/phase2-mortal-s42-v1/agent.json" \
  --qualification-run runs/phase2-teacher-qualification-v1 \
  --run phase2-pilot-1000-v1 --device cpu
```

准备器核验正式教师资格、教师身份、全部四席 provenance、训练/评测种子隔离、两来源各 1,000 局与全局 90/5/5 分割。仅 train/validation 做原生动作重放；test 只核验 split 与 validation receipt 元数据。已有弱数据的旧 test receipt 缺少 provenance 时，其限制保留在 source audit 中，不修改旧数据。

输出在外置 `runs/phase2-pilot-1000-v1/`：

- `manifest.json`：九任务、来源与运行身份、资格证据和各文件 SHA。
- `{weak,teacher}-training-counts.json`：按局统计的有效训练样本数。
- `{weak,teacher}-source-audit.json`：来源、种子、receipt 和身份审计。
- `configs/{A,B,C}-seed{2026,2027,2028}.json`：九个完整配置。

| Job ID | 配置文件名 | 训练 run 名称 | 阶段与来源 |
| --- | --- | --- | --- |
| A-2026 | A-seed2026.json | phase2-pilot-1000-v1-a-2026 | 弱数据 BC |
| A-2027 | A-seed2027.json | phase2-pilot-1000-v1-a-2027 | 弱数据 BC |
| A-2028 | A-seed2028.json | phase2-pilot-1000-v1-a-2028 | 弱数据 BC |
| B-2026 | B-seed2026.json | phase2-pilot-1000-v1-b-2026 | 教师数据 BC |
| B-2027 | B-seed2027.json | phase2-pilot-1000-v1-b-2027 | 教师数据 BC |
| B-2028 | B-seed2028.json | phase2-pilot-1000-v1-b-2028 | 教师数据 BC |
| C-2026 | C-seed2026.json | phase2-pilot-1000-v1-c-2026 | 教师数据 GRP → CQL |
| C-2027 | C-seed2027.json | phase2-pilot-1000-v1-c-2027 | 教师数据 GRP → CQL |
| C-2028 | C-seed2028.json | phase2-pilot-1000-v1-c-2028 | 教师数据 GRP → CQL |

公平性以完整准备配置核验：三臂共享 74×34 QNetwork、相同模型大小、batch 64，且同一个训练 seed 的初始 Q 权重相同。策略更新数均为 `U = ceil(3 × max(弱数据有效决策数, 教师有效决策数) / 64)`；这是相同优化更新/样本预算，较小来源可能重复更多轮。BC 冻结不参与 CE 的 v/aux 头；C 的 GRP 为教师训练局一轮、batch 32，其成本另记。所有标签仍由原始 trace 严格验证，筛选依完整教师内容身份，不按座位编号。

## 有限训练批处理

训练批处理先绑定现有 pilot 的真实 PID、创建时间和 argv，等待其进程退出且终态为 `prepared-not-trained`；随后重新核验九份完整配置。登记只保存不可变合同，`run` 才开始等待和调度。以下命令用于首次登记/启动；实际进度以主进度文档及外置回执为准，不重复已完成的登记或已开始的 run。

```sh
run_phase2_training_batch() {
  "$PHASE2_PYTHON" -B -u "$PHASE2_BATCH_TOOLS/scripts/phase2_training_batch.py" \
    --runtime-root "$PINNED_ROOT" --tools-root "$PHASE2_TOOLS" "$@"
}
run_phase2_training_batch register \
  --run phase2-training-1000-launch-v1 \
  --max-concurrent 2 --min-available-memory-bytes 2147483648 \
  --max-wall-seconds 172800
run_phase2_training_batch run --run runs/phase2-training-1000-launch-v1
```

上例显式允许最多两个不同 job；省略 `--max-concurrent` 时仍为串行。**A-2026 必须先独占完成整个 BC 预算**，由原 runner 的实际 `max_rss_bytes` 得到 R。之后每次派发重新读取 `psutil.virtual_memory().available`：空闲时至少需要 `max(2 GiB, R)`，派发第二个 job 至少需要 `max(2 GiB, 2R)`。第二个条件不足时保留现有 job 继续，退回串行；没有活动 job 而单任务门槛仍不足时，返回 `paused-memory-needs-resource-review`，不无限等待。登记时或过去测到的可用内存不能替代这些现场快照。

批处理固定 **九个训练任务、十五个命令步骤**：六次 BC，以及每个 C 的 `inspect → GRP → CQL` 三步。C 从原 runner inspection 得到 GRP 目标 G；只有 GRP 返回 `rc=2`、`status=interrupted`、`stop_reason=update_limit` 且 `progress={grp:G,cql:0}`，才允许同任务显式 `--resume` 进入 CQL。这是唯一自动恢复分支，不适用于预算暂停或异常退出。

**整次 batch 的 48 小时包含等待 pilot、预检和所有训练步骤**；每个原 runner 获得当时剩余的全局预算。任何非预期返回、wall-time 暂停或输入变更都会停止派发；已经启动的正常任务收尾后保留证据。SIGTERM/Ctrl-C 则按真实父子关系和 PID 创建时间清理拥有的跨 session 进程，保留日志与原回执；强制清理或未封存尝试不能被写成正常成功。实际运行需要允许读取宿主进程表，以支持这项清理。

批处理目录保存 `registration.json`、完整进程身份的 `started.json`、`prepared.json`、`events.jsonl` 和每步的原 argv、stdout/stderr、退出码、耗时、实时内存及原 runner `finished.json` 的 SHA。九任务结束后重新验收**当前全部模型**及同 seed 三臂初始化，生成 `completed-models.json` 并绑定到最终 `result.json`。只有 `status=complete` 才是可交接的训练终态。

同一 batch 不能重启或续领 48 小时；不会自动处理预算暂停，也不会在失败后改配置重跑。下文单任务命令用于检查和明确决定后的独立恢复，不与正在运行的 batch 同时执行。

## 有限开发评测批处理

开发批处理登记要求训练 batch **已实际启动**并有真实 `started.json` 进程身份，或者已有对应的成功终态；仅创建训练 registration 不够。可以在训练等待 pilot 时，另一个终端重复本页变量设置后登记开发 batch，让它等待；也可以在训练成功后登记。

```sh
run_phase2_development_batch() {
  "$PHASE2_PYTHON" -B -u "$PHASE2_BATCH_TOOLS/scripts/phase2_development_batch.py" \
    --runtime-root "$PINNED_ROOT" --tools-root "$PHASE2_TOOLS" \
    --training-tools-root "$PHASE2_BATCH_TOOLS" "$@"
}
run_phase2_development_batch register \
  --training-batch runs/phase2-training-1000-launch-v1 \
  --matrix-name phase2-development-small-v1 \
  --run phase2-development-1000-launch-v1 \
  --max-concurrent 2 --min-available-memory-bytes 2147483648 \
  --max-wall-seconds 172800
run_phase2_development_batch run --run runs/phase2-development-1000-launch-v1
```

开发 batch 等待原训练进程退出，复核成功终态、十五步原始回执、C 阶段顺序、无异常清理及 `completed-models.json`，再重新验证九个当前模型。通过后才调用原 matrix CLI 创建 `small` 矩阵。

范围固定为 **20 个 campaign、共 20,000 局**：九模型和 incumbent 各在 primary/weak 两池运行一次；每个 campaign 使用 2660000–2660249 的 250 个 seed、四次座位轮转，共 1,000 局。Primary 对手为 teacher/incumbent/efficiency；weak 为 efficiency/random/tsumogiri。先依次独占运行完整的 `primary/incumbent` 与 `primary/A-2026`，它们属于这 20 个 campaign，不额外增加样本。两者通过后才允许最多双 campaign。

开发内存门禁仍有 2 GiB floor；R 改为两个串行 primary campaign 中较大的**采样进程树 RSS 峰值**，单/双任务分别要求 `max(2 GiB,R)` / `max(2 GiB,2R)`。采样是命令及后代 RSS 的和，共享页可能重复计算、短时峰值可能漏采，不等于训练 runner 的 OS maximum RSS。每次派发仍检查当时真实 available；不足时降串行，空闲且仍不足则暂停资源审核。

每个 campaign 必须原始封存完整、身份/种子相符、failed/unsealed/illegal actions 均为 0，才计入完成。全部 20 个 campaign 通过后才调用 `report-development`，从完整配对组重算统计并核对报告。开发目录保留每步原日志、命令、进程身份、退出码、封存 SHA、采样 RSS 和最终 `development_report` 绑定。

开发 batch 自身也只有**一次含等待训练在内的 48 小时预算**，不会把等待时间排除后再增加 48 小时；不自动 resume campaign，不自动重试失败或新增种子。成功后停在报告与扩量判断：`scale_10000_eligible` 只说明登记的教师数据臂改善门槛是否达到，仍需结合成本审阅。不会自动采集 10,000 局、冻结最终候选、消耗 final seeds、运行 Akochan 或上线 Bot。

两个 batch 都以退出码 0 表示正常登记/完整成功，2 表示明确暂停，其余非零表示失败；必须同时读取 `result.json` 的真实状态。不要仅凭目录、进程消失或旧日志末行推断完成，也不要并行启动单任务命令绕过 batch 的预算和并发门禁。

## 单任务检查和底层调用

定义一个只运行显式指定任务的命令；它没有内部并发调度。

```sh
run_phase2_job() {
  "$PHASE2_PYTHON" -B -u "$PHASE2_TOOLS/scripts/run_prepared_training.py" \
    --runtime-root "$PINNED_ROOT" --preparation "$PHASE2_PREPARATION" \
    --max-wall-seconds 172800 "$@"
}
run_phase2_job --job A-2026 --inspect
run_phase2_job --job C-2026 --inspect
```

`--inspect` 只读复核准备证据并返回 `targets`、设备和实际任务名，不创建训练目录。C 的 `targets.grp` 是第一段的更新数 G，`targets.cql` 是 U。三个 C 任务的 G 相同。

六个 BC 任务的实际调用如下，每条结束后检查退出码和 `finished.json` 再运行下一条：

```sh
run_phase2_job --job A-2026
run_phase2_job --job B-2026
run_phase2_job --job A-2027
run_phase2_job --job B-2027
run_phase2_job --job A-2028
run_phase2_job --job B-2028
```

C 分两次调用，使 GRP 与 CQL 的端到端成本分别可见。下面从已经核验的准备文件计算 G，不打开任何 trace；值应等于 inspection 的 `targets.grp`。

```sh
PHASE2_GRP_UPDATES=$("$PHASE2_PYTHON" -B -c '
import json, math, sys
from pathlib import Path
p = Path(sys.argv[1])
c = json.loads((p / "configs/C-seed2026.json").read_text())["grp"]
rows = json.loads((p / "teacher-training-counts.json").read_text())
print(math.ceil(sum(r["grp_samples"] for r in rows) / c["batch_size"]) * c["epochs"])
' "$PHASE2_STORAGE/$PHASE2_PREPARATION")
run_phase2_job --job C-2026 --stop-after-updates "$PHASE2_GRP_UPDATES"
```

第一段正常应返回退出码 2、`status=interrupted`、`stop_reason=update_limit`，且回执内 `cli_result.progress` 为 `grp=G,cql=0`。确认这三个条件后启动第二段：

```sh
run_phase2_job --job C-2026 --resume
run_phase2_job --job C-2027 --stop-after-updates "$PHASE2_GRP_UPDATES"
```

同样检查 C-2027 的 GRP 段，再依次执行：

```sh
run_phase2_job --job C-2027 --resume
run_phase2_job --job C-2028 --stop-after-updates "$PHASE2_GRP_UPDATES"
```

检查 C-2028 的 GRP 段后执行 `run_phase2_job --job C-2028 --resume`。不要用无条件 shell 循环跳过阶段检查。

若 GRP 尚未完成就碰到 48 小时预算，读取最后回执的 `cli_result.progress.grp`，下次显式 `--resume --stop-after-updates <G 减去已完成 GRP 更新数>`，直到 `grp=G,cql=0`。`--stop-after-updates` 是本次调用额外允许的更新数，不是累计 checkpoint 位置。CQL 阶段或 BC 暂停则直接显式 `--resume`。

12 核、16 GiB 本机默认串行。仅在观察到实际 RSS 和系统内存压力有余量后，由外部调度同时运行最多两个不同 job；同一 job 有进程锁，禁止并行恢复。教师采集、资格或评测同时运行时，应在成本分析中注明负载，不能把墙钟时间描述为空闲吞吐。本脚本不实施全机并发上限。

## 暂停、恢复与失败

每次调用最大 172,800 秒。runner 从本次调用开始计时，将扣除自身预检的剩余预算传给冻结 CLI；CLI 再计入原生数据预检和恢复开销。训练在安全更新边界保存，因此最后一个更新/验证、导出和调用后验收可能跨过计时阈值。没有强杀，也不改变模型配置或恢复合同。

| Runner 退出码 | 含义 | 后续处理 |
| --- | --- | --- |
| 0 | `complete` 或共同门禁确认的 `already-complete` | 查看完成证据，可进入下一任务 |
| 2 | 安全暂停或显式更新上限 | 检查原因与 progress，下一次显式恢复 |
| 1 / 其他非零 | 验证或执行失败 | 先检查错误和原始 stderr，不当成成功 |

底层训练 CLI 可能对 `paused`/`interrupted` 返回 0；runner 解析真实结果并给出上表退出码。已存在训练文件但没有 `--resume` 会拒绝执行。恢复只能使用准备任务自己的 `checkpoints/<run>/latest.pt`，并绑定配置、数据、种子、原生计划、源码和实际运行时；冻结 CLI 进一步核验模型、优化器、RNG 和最佳快照。

runner 还要求已有 `metrics.jsonl` 与 checkpoint 更新位置完全一致。硬崩溃若留下领先 checkpoint 的日志，不能静默裁剪日志或重放掩盖重复更新；应保留故障证据，分析后另行登记处理。损坏的 complete 声明同样不能触发跳过：最终权重、版本化最佳权重、描述符、sidecar、训练计划和每次更新日志都必须通过共同门禁。

直接底层命令形状为：

```sh
"$PHASE2_PYTHON" -B -u -m agari_research.cli run \
  --config "$PHASE2_STORAGE/$PHASE2_PREPARATION/configs/A-seed2026.json" \
  --split datasets/manifests/selfplay-v1/split-v1.json \
  --run phase2-pilot-1000-v1-a-2026 --max-wall-seconds 172800
```

训练 batch 始终调用原 runner，保留下述调用证据；直接命令仅用于解释实际执行接口，不与 runner 或 batch 同时执行。

## 产物、成本与评测交接

每个 job 的执行证据位于外置 `runs/phase2-pilot-1000-v1-execution/<Job ID>/`。`binding.json` 固定准备证据、配置、split、实际运行时代码、Python、runner 和共同验收 helper 的 SHA。每次新调用创建不同的 `attempt-NNNN/`，包含 `started.json`、实际 `command.json`、完整 `stdout.log`/`stderr.log` 与 `finished.json`，不覆盖旧回执。没有启动子进程的预检暂停或 already-complete 调用自然没有子进程日志。

`finished.json.process` 使用 `wait4` 记录本次训练子进程的 wall/user/system 秒数和 `max_rss_bytes`；子进程数据包含原生预检、恢复和训练。`invocation_wall_seconds` 还包括 runner 自身门禁及调用后验证。原始训练 `runs/<run>/result.json` 的 `invocation_elapsed_seconds` 只覆盖 `run_training`，不能当成整个命令时间。C 的 GRP 段和 CQL 段分别保留完整成本，每段重复的数据预检/恢复开销必须明确包含在内；这些记录不是纯 kernel 时间。

完成产物保持在原有三类目录：

- `runs/<run>/`：config、dataset、execution-plan、逐更新 metrics、coverage、TensorBoard 和最终 result。
- `checkpoints/<run>/latest.pt`：最终预算与全部可恢复状态，绑定最佳验证快照。
- `models/<run>/`：`bc.pt`，或 `grp.pt` 和 `cql.pt` 的 final；以及 `best-<method>-uNNNNNNNNN.pt`、sidecar、descriptor 和方便访问的 best alias。

所有验证使用完整 validation，间隔 `ceil(U/3)`，阶段末再验证。每种方法只用自己的 `validation_loss` 选最小值，完全相同时保留更早 update；BC 是 masked CE，CQL 是完整 CQL 目标。逐动作类型及 choice/forced 诊断保留，不能拿跨方法 loss 排名。best 可以早于 final，但只有所有阶段完成后才有资格进入评测。

执行端和评测端共用 `phase2_evaluation_matrix.py::verify_completed_job(rt, prepared, job_id, job)`。它返回实际版本化最佳权重及 SHA、完整训练证据、初始 Q 身份和固定推理参数。评测矩阵还要检查同一 seed 三臂初始 Q SHA 相同、九模型均完整，随后复制并绑定 `latest.pt` 选定的不可变快照。`best-bc-agent.json` / `best-cql-agent.json` 便于浏览，但正式评测不能手选其他 update 或仅信任可被恢复重写的 alias。最后跨臂的候选选择以登记的 development 对战指标进行；开发评测由上面的独立 batch 执行，两套流程都不消费训练数据 test 分区的动作。
