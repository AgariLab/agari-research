# 第二次连接中断：已开始的正式排位未收到终局

`runs/riichi-ranked-continuation78-v1/` 于 `2026-10-28T02:25:27.405569+00:00` 停止，父记录保持 `stopped / game_incomplete`。本批 7 次尝试中有 6 场完整完成；第 7 次已收到 seat 2 的 start_game，之后记录 115 个 decision 和 115 个 accepted ack，但没有 end_game，最终为 `interrupted / connection_closed`。这是累计第 30 次尝试、第二次中断，不是完整场次或最终验收。

## 已记录事实

| 项目 | 结果 |
| --- | --- |
| 本批 requested / attempted / completed / healthy | 78 / 7 / 6 / 6 |
| 失败尝试 | `game-000007`，总 attempt 30，seat 2 |
| 耗时 | 313.85760312504135 秒 |
| 事件 | 231 行：1 start_game + 115 decision + 115 action_ack；无 end_game |
| 请求 / 准备 / 发送 / accepted | 均为 115，request_id 唯一且逐一对应 |
| 记录中的降级、超时、拒绝、默认代打等计数 | 均为 0；仅涵盖本地仍有观测的阶段 |
| 本地终分 / 最终顺位 / 具体评级变动 | 未知；scores=null |
| close code / close reason / 更底层异常 | 未记录 |

父索引为本次尝试保留 `game_incomplete`、`game_failed`、`transport_incomplete` 三个错误。断线前的 115 次 accepted 不能抵消断线，也不能把这次尝试认定为完整或健康。

## 完整场统计与全部观察量分开

| 范围 | 完整场数 | 中断尝试数 | 完整场请求数 | 全部已观察请求数 |
| --- | ---: | ---: | ---: | ---: |
| 本批 continuation78 | 6 | 1 | 944 | 1,059 |
| 累计：首场 + 原 remaining99 + 本批 | 28 | 2 | 4,367 | 4,482 |

累计共有 30 次已结束的客户端尝试。两种请求统计相差 115，恰为本次未完成尝试的请求数。这 115 次请求可用于连接故障和传输分析，**不能并入完整场的得分、顺位或完整场动作统计**。第一次中断没有记录动作，其独立证据继续保留在 [首次中断目录](../connection-interruption-v1/)。

## 断线后的平台行为与未知结果

官方 ranked 文档说明：对局中的 WebSocket 连接断开后，服务器会为余下动作使用默认动作，对局继续完成，结果仍计入 bot 的评级。因此本次不能表述为“未完局所以不计 rating”。本地没有收到终分，尚未取回该场的服务器结算记录；具体最终得分、顺位和评级变化仍未知。[官方 Ranked Matches 文档](https://riichi.dev/docs/ranked)（核对日期：2026-10-28）

本地 `ack_defaulted=0` 只描述断线前收到的回执，不代表断线后服务器没有默认代打。也不能把客户端尝试数直接当作已核实的服务器结算场数。

## 实现身份与诊断边界

七份原件中的本地引擎仍为 RiichiEnv 0.4.10，二进制 SHA 为 `169c8c2c53bdace01e4ca319f54050565fe728a5b1b2fc6b7f3046738887071d`；模型权重、工具源码、研究源码和 supervisor 摘要见 `summary.json`，与该运行 manifest 相符。

以下是主代理在事故处理会话中报告的执行观察，不是这七份原件中的字段：监督 session 83479 退出码为 1，PID 7975/7983 已不存在；公共 `/status` 当时报告服务器引擎 0.4.11，直连 TLS 探测超时、经代理访问 status 可达。服务器升级时间和断线原因均未确定；这些观察不能证明引擎版本差异或代理导致本次中断。归档时尚未启动第三批。

一次中断方案的 [audit_ranked_continuation.py](../audit_ranked_continuation.py) 保持原 SHA `c45a93bbac71db060dbaf0cf8436288eaa3458a380081e938aafea5575415cd2`，对当前事实返回 invalid，包含 `continuation_stopped_or_failed` 和 attempt 30 未完成错误。原 `audit_ranked.py` 也保持冻结；没有放宽任何既有通过条件，没有生成最终验收或模型发布结果。

## 原件与完整性

只读来源：`/Volumes/新加卷/AgariLab/runs/riichi-ranked-continuation78-v1/`。

- `parent/`：父批次 manifest.json、summary.json、games.jsonl。
- `failed-game-000007/`：失败场 manifest.json、report.json、games.jsonl、events.jsonl。
- 顶层 `summary.json`：从原件重算的派生摘要，区分本地事实、官方规则和执行会话观察。

七份原件共 40,553 bytes，逐字节复制并与外置源复核。JWT 形状和 PEM 私钥头静态模式检查均为 0 命中；模式检查不等于通用秘密检测。未读取凭证或环境值，未连接游戏服务器，未改模型、源码、运行环境或顶层实验计划。

在本目录核验全部原件及说明：

```sh
shasum -a 256 -c SHA256SUMS
```
