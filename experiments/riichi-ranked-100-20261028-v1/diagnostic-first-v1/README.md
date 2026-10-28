# 首次单场传输诊断：完整结束，未复现局中断线

`runs/riichi-ranked-diagnostic-v1/` 于 2026-10-28 02:53:41–02:57:18 UTC
完成一场正式排位。父批次与子对局均为 `complete`，父索引为 `healthy=true`、
`errors=[]`；诊断结果为 `finalized`、`error=null`。本次只尝试一个连接、创建一个连接，
没有自动重连，也没有再次 validation。

## 完整对局证据

| 项目 | 原件中的结果 |
| --- | --- |
| requested / attempted / completed / healthy | 1 / 1 / 1 / 1 |
| 座位、终分 | seat 0；`[4600, 32500, 49800, 13100]`，第四名 |
| 请求 / 准备 / 发送 / accepted | 均为 127 |
| 游戏事件 | 256 行：1 start_game + 127 decision + 127 action_ack + 1 end_game |
| 超时、拒绝、默认代打、陈旧或未知回执、降级 | 所有已记录计数均为 0 |
| 决策耗时 | 均值 7.109660775342032 ms，最大 26.78370801731944 ms |
| WebSocket 成功返回的应用消息 | 1,049 条；不同于记录决策与回执的事件数 |
| 库最后报告的 ping 延迟 | 0.2906494999770075 秒 |

127 个 decision 的 request_id 均唯一，与 127 个 accepted 回执逐一对应，事件顺序为
decision/ack 成对出现。唯一 end_game 是最后一条游戏事件；终分与子游戏记录、父索引完全一致，
四家分数合计 100,000。诊断没有 `recv_failed`、`send_failed` 或 `handshake_failed`，
也没有观察到库的本地 keepalive 超时签名。

## 完局后的关闭记录与解释边界

以下秒数为诊断事件相对于观察器启动的单调时钟偏移：

| 事件 | elapsed_seconds | 距最后一次 recv | close frame |
| --- | ---: | ---: | --- |
| recv_cancelled | 216.9969409169862 | 0.0004865420050919056 秒 | 收、发均未记录 |
| connection_lost | 217.00494208396412 | 0.008503457996994257 秒 | 收、发均未记录 |
| context_exit | 217.1670564999804 | 0.1706379579845816 秒 | 收、发均未记录 |

`connection_lost` 和 `context_exit` 的 `abnormal_no_close_frames=true` 如实保留。
它表示连接关闭时未记录 WebSocket Close frame，不能直接当作本局未完成或不健康。
`connection_lost` 的异常类别、接收异常类别，以及 context 退出异常类别均为 null；
整个阶段 `local_keepalive_timeout_observed=false`。

原 transport 在 ranked 的 end_game 被处理后结束主循环，先取消接收任务并清理 worker，
随后才退出 WebSocket context。结合真实 end_game、上述顺序及完整回执，本次关闭发生于
已完成对局后的清理阶段，不影响本局完整完成结论。

**关闭主动方未知，不能断言是客户端主动 abort。** 现有记录也兼容清理期间来自远端或代理的 EOF。
本次没有复现前两次缺失 end_game 的中断，不能证明旧问题已修复，也不能将旧中断归因于
keepalive、代理、服务器引擎版本或这次完局后的关闭现象。

## 实现身份与后续计数

本次使用独立的 `parent_transport_observer` 运行变体；诊断 manifest 明确记录
`same_runtime_as_uninstrumented=false`。模型、推理源码和原连接配置保持固定，
但不能把带观察器的运行称为与旧运行完全相同的传输 runtime。

- wrapper SHA256：`9c8dd327662276771082a5dd543b7bd4b00d8fa419ce3de64962a808cd10e294`。
- 原 supervisor SHA256：`cd570e8abc1b5b028661f6dfa7881ebdea7c650db11b7d5a0c7fd53adbe4aede`。
- toolkit source SHA256：`e1991979d79a373ca54050786616f57ded571a8d20de326a6e66982244dd1979`。
- research source SHA256：`43eabb39e1518ec550f8692bce79224060f07aa280f12a254b990ccc2c3e7e21`。
- PPO 权重 SHA256：`613d6a20cf1057a9c6556153a5506fdc11d115f62aaf5de83e39143572bd9096`。
- 本地引擎 RiichiEnv 0.4.10；Python 3.12.16、websockets 15.0.1；环境代理路径。
- ping interval/timeout 保持 20/20 秒；open/close timeout 为 15/2 秒，max_size 为 2,097,152，
  max_queue 为 16；没有调整超时策略。

此前累计 30 次客户端尝试、28 个完整局和 2 次中断；加上本次为 **31 次尝试、29 个完整局、
2 次中断**。完整场请求数由 4,367 增至 4,494；全部已观察请求数由 4,482 增至 4,609，
两者仍相差第二次中断场的 115 次。失败尝试不计作完整场，也不从历史中删除。

[后续预登记计划](../observed-series-plan-v1.json)限定最多 71 个独立的新 run，每个 games=1，
同时最多一个连接，首个不成功 run 后立即停止、不重试失败 run、不重复 validation。
只读核对时计划 SHA256 为 `554a8c1d9d8169f01344066e20351ac11f8ac78352f24cb77c66b68fade96d41`；
71 个 run 名唯一且连续，计划所列实现摘要、代理路径和 20/20 参数与本次原件一致。
若这 71 次均完整且没有额外尝试，累计将为 102 次客户端尝试、100 个完整局、2 次中断。
这只是执行上界与计数推导，未预先接受后续结果；原连续零故障标准仍为 false。

## 原件、来源与完整性

只读来源为本机外置目录 `/Volumes/新加卷/AgariLab/runs/riichi-ranked-diagnostic-v1/`。
本目录包含十份逐字节原件，共 **42,179 bytes**：

- `parent/`：父 manifest.json、summary.json、games.jsonl。
- `game-000001/`：子 manifest.json、report.json、games.jsonl、events.jsonl。
- `diagnostic/`：观察器 manifest.json、events.jsonl、result.json。

[SHA256SUMS](SHA256SUMS) 覆盖十份原件及本 README。复制后重新逐字节比对外置来源；
没有改写原状态、诊断布尔值或旧失败证据，没有生成整批最终验收或模型发布结果。
JWT 形状、PEM 私钥头、凭证值前缀、凭证赋值和常见服务密钥的静态模式扫描均为 0 命中；
模式扫描只覆盖所列格式，不等于通用秘密检测。归档没有读取凭证文件或环境值，
没有连接平台，也没有改动冻结脚本、模型、运行环境或后续计划。

从本目录核验：

```sh
shasum -a 256 -c SHA256SUMS
```
