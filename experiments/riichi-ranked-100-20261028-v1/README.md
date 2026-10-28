# 首批 100 场正式排位

2026-10-28：用户提供 Bot Token 后，真实验证已通过，用户确认 Bot 状态 active。
验证只进行一次，证据见 [validation](../riichi-validation-20261028-v1/README.md)。
本批目标为 100 场完整正式半庄，检验线上接入与运行可靠性，不作为棋力晋级验收。

## 固定条件与执行方式

- 始终使用 `models/full-coverage-trace-1000-v1/ppo-agent.json`，PPO SHA-256：
  `613d6a20cf1057a9c6556153a5506fdc11d115f62aaf5de83e39143572bd9096`。
- 真实端点 `wss://game.riichi.dev/ws/ranked`，RiichiEnv 0.4.10，默认 Tenhou 四人赤牌半庄。
  本机引擎二进制和推理实现逐场绑定；服务端未声明其二进制摘要，兼容性通过真实请求检验。
- 首场使用现有 CLI 单场运行，目录 `runs/riichi-ranked-first-v1/`。
  首场健康后，剩余 99 场使用 `agari-kit/scripts/run_online_campaign.py`，
  工具提交 `e086d9f308736a5422d4a63ace35f07b5d90eeb9`，脚本 SHA-256：
  `cd570e8abc1b5b028661f6dfa7881ebdea7c650db11b7d5a0c7fd53adbe4aede`。
- 剩余运行目录 `runs/riichi-ranked-campaign-remaining99-v1/`，每场独立子目录；
  每场完整结束后再重连排队，禁止同一 Bot 的并行连接。
- 对手由服务器匹配，座位及随机牌山由服务器决定；不因顺位或得失点停止、剔除或更换模型。
  所有完成及失败记录均保留。若出现协议、身份或运行健康问题，在该场结束或连接失败后停止续排。

监督门槛为：真实 `end_game`、无断线失败、无降级；请求数、发送数和 `accepted` 回执数一致且非零；
超时、`rejected`、`unparseable`、`defaulted`、`stale`、未知回执和降级后抑制动作均为零。
模型、引擎、工具源码和监督脚本身份逐场核对。等待匹配的空闲超时为 1,800 秒，单次动作仍按服务器预算执行。
脚本的 10 项模拟测试覆盖停止门槛、断线、旧目录保护、身份变化、凭证不入日志及取消清理。

初始批次启动命令（在研究仓库，现已停止，原目录保留）：

```sh
HTTPS_PROXY=http://127.0.0.1:10800/ HTTP_PROXY=http://127.0.0.1:10800/ \
python3 ../agari-kit/scripts/with_storage.py uv run --frozen python ../agari-kit/scripts/run_online_campaign.py \
  --agent '@/Volumes/新加卷/AgariLab/models/full-coverage-trace-1000-v1/ppo-agent.json' \
  --token-file ../agari-kit/.riichi-token --games 99 \
  --idle-timeout 1800 --reserve-ms 100 --run riichi-ranked-campaign-remaining99-v1
```

## 当前记录

首场已完整完成：seat 0，终分 `[-300, 14600, 46100, 39600]`，第四名；
127 次请求、发送和 `accepted` 回执，零超时、非法回执或降级。原始安全记录保存在 `first-game/`。
原 99 场监督运行于 2026-10-28 01:37:48 UTC 停止：21 场完整，随后第 22 次尝试约 33.53 秒后
`connection_closed`；events 为空、请求数为 0、seat/scores 为 null，未观察到 start_game。
首场加已结束的 21 场累计 22 场完整，3,423 个请求均 accepted，另保留这一次中断。
旧清单保持 stopped，不能声称原 99 场批次完成或连续零故障。
日志未保留 close code，具体关闭原因不明；没有本地开局记录不等于可以断言服务器未开局或评级未受影响。
[中断证据](connection-interruption-v1/README.md)保留原父索引、失败子记录及完整校验和。

确认旧进程真实退出后，于 01:42:37 UTC 使用同一源码、模型、引擎和监督脚本，在新目录
`runs/riichi-ranked-continuation78-v1/` 继续剩余 78 场。上文命令仅替换为
`--games 78 --run riichi-ranked-continuation78-v1`，未再次 validate，已收到 start_game 和 accepted 回执。
当前总完成数为新批次 summary 的 `games_completed + 22`；总连接尝试为 `games_attempted + 23`。

[原独立验收工具](AUDIT.md)及[原发布顺序](../full-coverage-trace-1000-v1/release-v2-workflow.md)
保留原门槛，发生中断后不能借其发布 `final-audit/` 或清单；历史 v1 不改。
恢复后的独立审核将另外保留全部 101 次尝试：100 场完整结果与 1 次中断分别展示，
只可记为 completed_with_interruption，不能冒充原连续零故障批次。
新的 continuation 审计与发布工具正在准备；续跑若再次失败，当前恢复方案也不能发布。

本地 `decision.latency_ms` 计消息接收到动作准备的客户端处理时间，
`action_ack.elapsed_ms` 是服务端报告的耗时，二者不混算。排位顺位是在线匹配结果，
不与固定种子、固定对手的离线筛选直接配对比较。
