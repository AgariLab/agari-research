# Mortal-S42 教师资格：正式封存结果

Mortal-S42 通过已登记的固定对手池资格门禁。双方各完成 1,000 局，覆盖相同的 250 个 seed，每个 seed 各四次座位轮转；双方 failed、unsealed 和 illegal actions 均为 0。教师相对 incumbent 的平均顺位差为 **−0.921**，配对 95% CI 为 **[−0.978, −0.861]**，上界低于 0。

这是教师面对三个固定 incumbent 对手的资格结论，可用于后续教师数据试验。它不代表对任意对手的全局棋力、Elo、线上段位或学生晋级；本阶段也没有通过资格对局评估学生。独立最终晋级样本门槛仍为每模型 10,000 局，本次报告的 `promotion_sample_size_met` 为 false。

## 完整比较

Baseline 为 incumbent 对阵三个相同 incumbent；teacher 为 Mortal-S42 对阵相同的三个 incumbent。规则为 Tenhou 四人红牌半庄，RiichiEnv 0.4.10。Seed 范围固定为 **2610000–2610249**，未筛选中途结果。

| 指标 | Baseline | Teacher | Teacher − baseline | 配对 95% CI |
| --- | ---: | ---: | ---: | ---: |
| 平均顺位，越低越好 | 2.500 | 1.579 | −0.921 | [−0.978, −0.861] |
| 平均得点变化，相对 25,000 点 | 0.0 | +16,018.9 | +16,018.9 | [+14,867.9, +17,213.3] |
| 一位率 | 25.0% | 61.1% | +36.1 个百分点 | [+32.7, +39.3] 个百分点 |
| 二位率 | 25.0% | 23.6% | −1.4 个百分点 | [−4.1, +1.4] 个百分点 |
| 三位率 | 25.0% | 11.6% | −13.4 个百分点 | [−15.2, −11.3] 个百分点 |
| 四位率 | 25.0% | 3.7% | −21.3 个百分点 | [−22.5, −20.0] 个百分点 |

CI 由冻结 `compare_evaluations.py` 重算：以 **250 个完整配对 seed 组**为抽样单位，四次座位轮转保留在同一组内，percentile bootstrap 2,000 次、bootstrap seed 0。资格主指标是平均顺位差；其余为同一完整比较的描述性结果，不能当作六项独立资格检验。精确数值见原字节 [qualification.json](qualification.json)。

## 来源与只读复核

正式原件在外置目录 `/Volumes/新加卷/AgariLab/runs/phase2-teacher-qualification-v1/`。只有两侧封存完成、正式 `qualification.json` 出现后才读取结果。2026-10-29 02:27:51（Asia/Shanghai；UTC 2026-10-28 18:27:51）使用原冻结 `phase2_contract.require_qualification` 独立复核，未调用会写报告的 `report` 入口，也未重新运行任何对局。

复核重新验证教师权重/runtime manifest、适配器源码、incumbent 权重、规则和引擎二进制，重建注册信息；验证所有组回执及汇总封存，再重算配对 CI，并要求与正式报告完全一致。绑定的原生合法性报告包含七种特殊事件测试、四席完整原生半庄及每席 trace/mjai 证据，共 16 个检查项。原件读取前后 SHA 全部一致。

| 证据 | SHA-256 |
| --- | --- |
| 正式 qualification | `13b1805c13b0957eea9e22044e43418f5c9ed9aa145018e88554f5135ee6f538` |
| registration | `65b87b00fc62eda020ee92e16be4004ff6170828990feb3621c4354d956d811f` |
| comparison contract 原文件 | `30c81aad0a2dcb109cbe07f11d9ac5cea604d2a231f52a661dd43cfa1603e343` |
| Baseline completion | `555fd81476a3aabc26aad119de70ac71e82a18c50a4532936017d146e14de559` |
| Teacher completion | `8ab2b74a7821284794d5fcde502805b74aa92ab8bc8f0250257a76490773a2ee` |
| 教师权重 | `e5f6955b6f19007fd65315e53291b36994493046bc118b991b959e9dc596d660` |
| 冻结 kit source | `c3a791b966a688a211bbb7bf1b87dcb908a80b474b75105ae90052d2d9a5cfb7` |
| 引擎二进制 | `169c8c2c53bdace01e4ca319f54050565fe728a5b1b2fc6b7f3046738887071d` |

运行时代码来自 `phase2-runtime-20261029-v1`：kit 提交 `78719f35ea17d68e3a6fcfd04a2aa969a95142cc`、research 提交 `80d3c4204d381a93a7f53d729fa59c6074ff1b42`。验证后两仓均保持干净。[verification.json](verification.json) 保存独立复核回执；[evidence-sha256.json](evidence-sha256.json) 保存原件路径、大小、逐文件 SHA 和冻结协议/验证工具 SHA。原文件 SHA 与比较器使用的 canonical identity SHA 属于不同层级，不能互换。

本目录的 qualification、registration、comparison contract、双方 completion/report/process log 均为原字节副本。较大的 manifest、games、candidate-latencies、plan 及逐组回执保留在外置原目录；这里保留其绑定哈希，不复制完整数据。README、verification、timing 和哈希清单为本次整理产物，不冒充原运行报告。

## 实测时间与内存口径

| 测量 | Baseline | Teacher |
| --- | ---: | ---: |
| 整次命令 wall | 154.80 s | 4,394.64 s（约 73.24 min） |
| 整次命令 user / system | 124.34 / 8.66 s | 2,947.40 / 615.56 s |
| 各局循环耗时总和 | 99.824 s | 1,237.874 s |
| 候选决策数 | 166,300 | 161,397 |
| 候选 `act` mean / p95 | 0.119 / 0.164 ms | 7.149 / 11.607 ms |
| 候选 `act` max | 0.547 ms | 92.175 ms |
| macOS `time -l` 报告的 maximum RSS | 279,379,968 bytes | 372,719,616 bytes |

整次命令的 real/user/sys 和内存值来自原始 [baseline-process.log](baseline-process.log) / [teacher-process.log](teacher-process.log)，包含启动、身份检查、逐局模型/worker 初始化和结果封存等开销。maximum RSS 是 `time -l` 的原始字节统计，不是同时采样后把父进程与所有教师 worker 内存相加的测量。

`report.elapsed_seconds` 是每局 agent reset 后对局循环的耗时总和，排除初始化和最终封存。候选延迟计时包住完整 `agent.act` 调用：排除初始化和平台网络；**教师值包含本地适配、worker IPC 和推理**。原报告的 `latency_scope` 写有 `excludes loading and transport`，其中对 external transport 的表述不精确；这里按冻结代码说明真实范围，保留原报告字节不作修订。

两侧资格曾并行，CPU/MPS 恢复验收及其他工作也与本次运行重叠。因此这些数值代表有并发负载时的实测成本，不能当作空闲吞吐，也不能只用 `act` 时间或各局循环时间预测完整四教师采集成本。精确计时及口径见 [timing.json](timing.json)。

只读复核可在同一固定环境下通过 `phase2_contract.require_qualification(storage, output, protocol_path, teacher_descriptor)` 重做；它要求外置原始封存证据仍在。重新验证不需要启动 qualification runner 的 `baseline`、`teacher` 或 `report` 子命令。
