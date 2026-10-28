# 第四次中断：握手成功后未收到应用消息，连接无关闭帧结束

`runs/riichi-ranked-observed-s02-000021-v1/` 于 2026-10-28 09:20:55–09:21:57 UTC
执行一次正式排位连接尝试，父批次最终为 `stopped / game_incomplete`，子记录为
`interrupted / connection_closed`。这是全局第 **94 次客户端尝试**；截至本次停止，
累计 **90 场完整、4 次中断**，100 场完整目标仍未达到。

本计划此前 20 次均已完整完成。直接前序 `riichi-ranked-observed-s02-000020-v1`
的父 summary 为 `complete`、`games_completed=healthy_games=1`，请求、准备、发送及
accepted 均为 **188**；它不属于本次失败，本目录没有将其记录归入失败场。
累计计数为既有 73 次尝试（70 场完整、3 次中断）加本计划已执行的 21 次尝试
（20 场完整、1 次中断）；尚未执行的计划项不计作尝试或完成场次。

## 原件中的事实

| 项目 | 结果 |
| --- | --- |
| 本 run requested / attempted / completed / healthy | 1 / 1 / 0 / 0 |
| 父批次 / 子记录耗时 | 61.53692220791709 / 61.51904408307746 秒 |
| 诊断总耗时 | 61.57610837498214 秒 |
| 游戏事件 | `game-000001/events.jsonl` 为 0 字节；0 start_game、0 end_game |
| 请求 / 准备 / 发送 / accepted | 均为 0；无动作耗时样本 |
| 本地座位、终分 | `seat=null`、`scores=null` |
| 子记录 | `status=interrupted`、`failure=connection_closed`、`degraded=false` |
| 诊断结果 | `status=finalized`、`error=null`、`campaign_status=stopped` |
| 诊断数量 | connection_attempts=1、connections_created=1、events_recorded=7 |

以下秒数为诊断事件相对于观察器启动的单调时钟偏移：

| 事件 | elapsed_seconds | 主要字段 |
| --- | ---: | --- |
| diagnostic_ready | 0.49362024990841746 | — |
| connection_attempt | 0.49364424997475 | attempt=1 |
| connection_created | 0.49424420890863985 | connection_id=1 |
| handshake_complete | 2.996548124938272 | recv_count=0 |
| connection_lost | 61.429023833945394 | recv_count=0；exception/cause/receive_exception 类别均为 null |
| recv_failed | 61.429403416928835 | exception_category=connection_closed_error；cause_category=null |
| context_exit | 61.54335849999916 | exception_category=null；cause_category=null |

本次有明确的 **handshake_complete**：从 connection_attempt 到成功握手为
**2.502903874963522 秒**；握手成功至 connection_lost 为 **58.43247570900712 秒**。
后三个关闭阶段事件均为：

- `received_close_code=null`、`sent_close_code=null`、`received_close_first=null`。
- `abnormal_no_close_frames=true`。
- `local_keepalive_timeout_observed=false`。
- `recv_count=0`、`seconds_since_last_recv=null`。
- `library_latency_seconds=0.31702858302742243`。

上述诊断事件不属于游戏事件。recv_count 的定义是成功返回的应用消息数量，不包括
WebSocket Ping/Pong 控制帧。已安装的 websockets 15.0.1 在确认 Pong 时更新 latency；
记录中的非零库延迟因此提供了 Ping/Pong 往返证据，但不能恢复具体帧数、时间序列或
确定随后连接关闭的原因。

## 故障类别与未知边界

证据支持 **WebSocket 握手成功后、尚未观察到应用消息或开局时，连接未记录 Close frame
即结束**。这与[第三次中断](../third-connection-interruption-v1/)的 15 秒打开阶段超时不同；
本次没有 handshake_failed 或 timeout 类别事件，也没有本地 keepalive 超时签名。
不能仅凭约一分钟的时长认定某个代理或服务端的超时策略。

`connections_created=1` 本身仍仅表示连接对象创建；本次成功握手的依据是独立的
handshake_complete 事件。关闭事件没有收发关闭码，不能改写成收到某个具体关闭码。
connection_lost 的异常类别为 null 也不表示正常 WebSocket 关闭；recv_failed 记录了
connection_closed_error，且关闭帧缺失。连接的关闭主动方、具体网络环节及根因均未知。

diagnostic 的 `finalized / error=null` 表示观察器正常收尾，**不表示对局完成或连接健康**。
父索引保留 `game_incomplete`、`game_failed`、`action_receipts_mismatch`、
`transport_incomplete`。监督器在 `requests==0` 时也产生 action_receipts_mismatch；
本次它表示未满足非零请求及完整回执门槛，**不是实际收到错误回执**。
记录中的推理超时、发送超时、错误回执及降级计数均为 0，但这不消除连接失败。

本地没有收到 start_game，仍不能证明服务器没有创建对局、没有后续默认动作或评级
没有受影响。服务器是否开局、最终结果、顺位及 rating 变化均为 **unknown**。
本目录没有取得服务端结算证据，也不能用本次现象解释此前其他中断的根因。

## 固定身份

- wrapper SHA256：`9c8dd327662276771082a5dd543b7bd4b00d8fa419ce3de64962a808cd10e294`。
- supervisor SHA256：`cd570e8abc1b5b028661f6dfa7881ebdea7c650db11b7d5a0c7fd53adbe4aede`。
- toolkit source SHA256：`e1991979d79a373ca54050786616f57ded571a8d20de326a6e66982244dd1979`。
- research source SHA256：`43eabb39e1518ec550f8692bce79224060f07aa280f12a254b990ccc2c3e7e21`。
- PPO 权重 SHA256：`613d6a20cf1057a9c6556153a5506fdc11d115f62aaf5de83e39143572bd9096`。
- Python 3.12.16、websockets 15.0.1、本地 RiichiEnv 0.4.10。
- 运行变体 `parent_transport_observer`，`same_runtime_as_uninstrumented=false`；
  proxy_policy=environment、network_path=proxy，未保存代理地址或环境值。
- ping interval/timeout 为 20/20 秒，open/close timeout 为 15/2 秒；没有自动重连。

归档时重新核对 wrapper 和 supervisor 的实际字节摘要，与原件一致；未修改任何冻结
源码、连接参数、模型、锁文件、环境或后续计划，没有生成最终排位验收或 release-v2。

## 原件映射与完整性

只读来源为 `/Volumes/新加卷/AgariLab/runs/riichi-ranked-observed-s02-000021-v1/`。
本目录复制其**全部 10 份受控原件，共 9,316 bytes**：

| 外置 run 中的相对路径 | 本目录中的相对路径 |
| --- | --- |
| manifest.json、summary.json、games.jsonl | parent/ 下同名文件 |
| game-000001/ 下 manifest.json、report.json、games.jsonl、events.jsonl | 保留 game-000001/ 及全部文件名 |
| diagnostic/ 下 manifest.json、events.jsonl、result.json | 保留 diagnostic/ 及全部文件名 |

空的游戏 events 文件按原样保留。复制后逐文件比较源与副本的字节、大小和 SHA-256，
全部一致；[SHA256SUMS](SHA256SUMS) 覆盖 10 份原件及本 README。
JWT 形状、Bearer 凭证值、PEM 私钥头、凭证赋值及常见服务密钥静态模式扫描均为 0 命中；
仅输出数量，不打印匹配内容。静态模式扫描不等于通用秘密检测。

归档没有读取凭证文件或环境值，没有连接 Bot、运行 validation 或重启运行。
从本目录核验：

```sh
shasum -a 256 -c SHA256SUMS
```
