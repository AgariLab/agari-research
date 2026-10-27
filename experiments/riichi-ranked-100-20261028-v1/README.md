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

实际启动命令（在研究仓库）：

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
剩余 99 场监督运行已启动，当前尚未将其标为全部完成；以外置 `summary.json` 与逐场记录为实时依据。

本地 `decision.latency_ms` 计消息接收到动作准备的客户端处理时间，
`action_ack.elapsed_ms` 是服务端报告的耗时，二者不混算。排位顺位是在线匹配结果，
不与固定种子、固定对手的离线筛选直接配对比较。
