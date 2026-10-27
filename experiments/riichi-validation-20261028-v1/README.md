# 首次 riichi.dev 真实平台验证：已通过

2026-10-28，`full-coverage-trace-1000-v1` PPO 模型完成首次真实平台验证。服务器返回 `validation_result.passed=true`；用户随后确认 bot 为 **active**。本目录只归档既有结果，没有再次连接游戏服务器或重复 validate。

原始记录来自 `/Volumes/新加卷/AgariLab/runs/riichi-validation-v1/`。`manifest.json`、`games.jsonl`、`events.jsonl`、`report.json` 均逐字节复制，外置原文件只读，合计 33,660 bytes。`summary.json` 为派生统计与核验结果，`SHA256SUMS` 覆盖四份原始副本及两份说明文件。

## 结果与证据边界

| 项目 | 结果 |
| --- | --- |
| 模式 / 完成数 | validate，1 场完整东风局 |
| 席位 / 终分 | seat 0；`[49100, 5400, 22100, 23400]`，本场第 1 |
| 请求 / 已准备回复 / 已发送回复 | 120 / 120 / 120 |
| 服务端 accepted ack | 120，逐一对应 120 个唯一 request_id |
| degraded / 本地超时事件 | 0 / 0 |
| defaulted / rejected / unparseable / stale ack | 均为 0 |
| events.jsonl | 243 行：1 start_game + 120 decision + 120 action_ack + 1 end_game + 1 validation_result |
| 时间银行 | 120 次 ack 的 bank_consumed_ms 均为 0；bank_ms 均为 15,000 |
| 动作类型 | dahai 85、none 31、reach 2、hora 2 |

官方验证规则为：seat 0 对阵平台生成的三个摸切 bot，完整结束东风局，期间不断线且不受 chombo 罚则，即自动激活；超时本身可以由服务器代打默认动作而仍通过。本场所有 ack 均为 accepted，没有默认代打记录。active bot 不应重复调用只供 pending bot 使用的验证入口。[官方 Bot Validation 文档](https://riichi.dev/docs/validation)（核对日期：2026-10-28）

这是一次连通、协议及完整完局验证，**不证明棋力**。终分及第 1 名仅描述本场对三个摸切 bot 的结果，不能外推为 ranked 表现。这里保存的是受限的传输摘要，不是完整 MJAI 牌谱；不断线和无 chombo 的通过结论以服务器 `validation_result` 为依据。

ranked 不属于本验证报告，排位进度及结果见 [独立 ranked 报告](../riichi-ranked-100-20261028-v1/README.md)。这里不记录会变化的排位累计数，也不据验证成绩推断排位结论。active 状态由用户确认，归档过程没有另行查询平台状态。

## 两种耗时不可混用

以下均直接由 120 条 decision 或 action_ack 计算，单位 ms。分位数采用排序后位置 `(n - 1) × p` 的线性插值；完整精度见 `summary.json`。

| 指标 | 本地 decision.latency_ms | 服务端 action_ack.elapsed_ms |
| --- | ---: | ---: |
| 最小 | 5.620 | 294.000 |
| 平均 | 6.321 | 342.625 |
| p50 | 6.397 | 300.000 |
| p95 | 6.582 | 665.250 |
| p99 | 6.688 | 817.810 |
| 最大 | 6.734 | 1012.000 |

本地 `latency_ms` 从客户端 `socket.recv()` 返回后记录的收包时刻算起，到响应已准备并记录 decision 为止。它包含客户端排队、消息解析、worker IPC/推理和合法动作检查；模型加载已在连接前完成，计时不包含 `socket.send()` 或网络往返。计时实现见 [connect.py](../../../agari-kit/src/agari_kit/connect.py) 与 [online.py](../../../agari-kit/src/agari_kit/online.py)。

`elapsed_ms` 是服务器 ack 原样提供的请求耗时。官方计时从服务器发送 `request_action` 开始，预算包含请求与回复的网络往返；它不是纯模型推理耗时，也不是客户端测得的接收 ack RTT。本记录不能进一步分离网络与服务端开销，不能把两列之差直接当作网络延迟。[官方 MJAI Protocol 文档](https://riichi.dev/docs/protocol)（核对日期：2026-10-28）

## 模型与实现核验

下列值已对当前实际文件重新计算，与原始 manifest 一致。模型权重、descriptor 和原生引擎没有复制进此小型证据目录。

| 对象 | SHA-256 |
| --- | --- |
| PPO 权重，701,520 bytes | `613d6a20cf1057a9c6556153a5506fdc11d115f62aaf5de83e39143572bd9096` |
| agent descriptor，219 bytes | `51495822b3af9c6aff6f11a59aa3db0c1380ed4238088b22d4bc2741e052a381` |
| agari-kit src，19 个 Python 文件 | `e1991979d79a373ca54050786616f57ded571a8d20de326a6e66982244dd1979` |
| agari-research src，17 个 Python 文件 | `43eabb39e1518ec550f8692bce79224060f07aa280f12a254b990ccc2c3e7e21` |
| RiichiEnv 0.4.10 原生模块 | `169c8c2c53bdace01e4ca319f54050565fe728a5b1b2fc6b7f3046738887071d` |

src 指纹按运行时实现复算：对排序后的每个 `*.py` 依次写入相对路径、NUL、文件内容、NUL；agari-research 路径含包根索引 `0/` 前缀。引擎身份通过既有 `agari-research/.venv` 只读查询。四份原始文件的 JWT 形状与 PEM 私钥头静态扫描均为 0 命中；这只是指定模式扫描，不是通用秘密检测保证。没有读取凭证文件或环境值。

在本目录执行 `shasum -a 256 -c SHA256SUMS` 可检查归档副本和说明文件的完整性。
