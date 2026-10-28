# 第三次中断：连接打开阶段超时，未观察到开局

`runs/riichi-ranked-observed-000042-v1/` 于 2026-10-28 06:46:24–06:46:39 UTC
执行一次正式排位连接尝试，最终父批次为 `stopped / game_incomplete`，子记录为
`interrupted / connection_or_worker_failed`。这是全局第 **73 次客户端尝试**；截至本次
停止，累计 **70 场完整、3 次中断**，100 场完整目标仍未达到。

此前 `riichi-ranked-observed-000041-v1` 已完整完成，父 summary 为 `complete`、
`games_completed=healthy_games=1`，请求、准备、发送及 accepted 均为 **229**。
它不属于本次失败，本目录也没有将其记录复制或归入失败场。
计数来自既有 31 次尝试（29 场完整、2 次中断）加本计划已执行的 42 次尝试
（41 场完整、1 次中断）；计划中尚未执行的 run 不计作尝试或完成场次。

## 原件中的事实

| 项目 | 结果 |
| --- | --- |
| 本 run requested / attempted / completed / healthy | 1 / 1 / 0 / 0 |
| 父批次耗时 | 15.62758037506137 秒 |
| 子记录耗时 | 15.61042362509761 秒 |
| 游戏事件 | `game-000001/events.jsonl` 为 0 字节；0 start_game、0 end_game |
| 请求 / 准备 / 发送 / accepted | 均为 0；无动作耗时样本 |
| 本地座位、终分 | `seat=null`、`scores=null` |
| 子记录 | `status=interrupted`、`failure=connection_or_worker_failed`、`degraded=false` |
| 诊断结果 | `status=finalized`、`error=null`、`campaign_status=stopped` |
| 诊断数量 | connection_attempts=1、connections_created=1、events_recorded=4 |

诊断事件使用相对于观察器启动的单调时钟偏移：

| 事件 | elapsed_seconds | 附加字段 |
| --- | ---: | --- |
| diagnostic_ready | 0.511809500050731 | — |
| connection_attempt | 0.5118267090292647 | attempt=1 |
| connection_created | 0.5124676671111956 | connection_id=1 |
| handshake_failed | 15.513273959048092 | exception_category=timeout、cause_category=timeout |

从 connection_attempt 到 handshake_failed 为 **15.001447250018828 秒**，与固定
`open_timeout=15` 相符。没有 handshake_complete、recv_failed、connection_lost
或关闭码事件。上述 4 条诊断事件不是游戏事件，不能计作收到开局或动作请求。

父索引保留 `game_incomplete`、`game_failed`、`action_receipts_mismatch`、
`transport_incomplete`。监督器在 `requests==0` 时同样产生 action_receipts_mismatch；
本次该字段表示未满足非零请求及完整回执的验收门槛，**不是实际收到错误或不匹配回执**。
记录中的 local_timeouts、send_timeouts、错误回执及降级计数均为 0；这些计数不能抹去
单独记录的连接打开超时，也不能说明未观测的服务器行为。

## 故障类别与推断边界

冻结 wrapper 在 `ObservedConnection.__init__` 中记录 connection_created，因此
**connections_created=1 只证明连接对象创建，不证明 TCP、TLS 或 WebSocket 握手成功**。
已安装的 websockets 15.0.1 将 create_connection 与随后 handshake 一起纳入
open_timeout；部分代理路径在 TLS 建立前便创建该对象。现有证据能定位为
**WebSocket 成功打开前的连接／握手阶段超时**，不能进一步确定具体代理、TLS 或
WebSocket HTTP 握手阶段，也不能确定故障责任方。

库只在握手成功后启动 keepalive。本次没有成功握手记录，不能归因为已观测到的
ping/pong 超时、模型推理超时或局中断线，也不能认定已经复现前两次中断的根因。
`finalized / error=null` 只表示诊断正常收尾，**不表示对局完整完成或连接成功**。

本地未观察到 start_game，仍不能据此断言服务器没有创建对局、没有产生默认动作或
评级没有受影响。服务器是否开局、最终结果、顺位及 rating 变化均为 **unknown**。
本目录没有取得服务端结算证据，没有推测 HTTP 拒绝状态或关闭原因。

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

归档时重新核对本地 wrapper 和 supervisor 的实际字节摘要，与原件一致。
本次没有修改已冻结实现、连接参数或既有通过条件，也没有生成最终排位验收或 release-v2。

## 原件映射与完整性

只读来源为 `/Volumes/新加卷/AgariLab/runs/riichi-ranked-observed-000042-v1/`。
本目录复制其**全部 10 份受控原件，共 7,879 bytes**：

| 外置 run 中的相对路径 | 本目录中的相对路径 |
| --- | --- |
| manifest.json、summary.json、games.jsonl | parent/ 下同名文件 |
| game-000001/ 下 manifest.json、report.json、games.jsonl、events.jsonl | 保留 game-000001/ 及全部文件名 |
| diagnostic/ 下 manifest.json、events.jsonl、result.json | 保留 diagnostic/ 及全部文件名 |

空的游戏 events 文件也按原样保留。复制后逐文件核对源与副本的字节、大小和 SHA-256，
全部一致；[SHA256SUMS](SHA256SUMS) 覆盖 10 份原件及本 README，不改写任何原状态。
JWT 形状、Bearer 凭证值、PEM 私钥头、凭证赋值及常见服务密钥静态模式扫描均为 0 命中；
只报告计数，不输出匹配内容。静态模式扫描不等于通用秘密检测。

本次归档没有读取凭证文件或环境值，没有连接平台、重启 Bot、改动环境或修改后续计划。
从本目录核验：

```sh
shasum -a 256 -c SHA256SUMS
```
