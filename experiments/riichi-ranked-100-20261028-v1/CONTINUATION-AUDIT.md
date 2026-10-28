# 中断后继续完成 100 场的独立审计

> 历史固定方案，当前不适用：continuation78 已在完成 6 场后再次中断。
> 本工具仍保留原门槛并拒绝发布，不得用于生成真实最终审核或 release-v2。
> 当前全部尝试和后续诊断进度见[排位记录](README.md)。

原 remaining99 监督运行已停止：21 场完整，随后第 22 次尝试记录 `connection_closed`。加上独立首场，共 22 场完整；原 batch 的全部 22 次尝试继续保留。失败尝试是总第 23 次，耗时约 33.53 秒，events 为空，seat/scores 未知。没有记录 close code 或异常详情；没有观测到 start_game 不能证明服务器未开局，也不能证明评级未受影响。

后续运行是独立的 `runs/riichi-ranked-continuation78-v1/`，请求 78 场，manifest 的启动时间固定为 `2026-10-28T01:42:37.069470+00:00`。本方案审核 **首场 1 + 原 batch 22 + 后续 78 = 101 次尝试，其中 100 场完整、1 次已记录中断**。它不改变原“无故障完成”门槛，旧 `audit_ranked.py` 与原 `final-audit/` 保持不动，原无故障结论仍不成立。

本工具只读取本地证据及固定模型/实现的身份，不连接游戏服务器，不读取凭证或环境值，不加载模型。当前运行中的场次不会作为完工记录读取。

## 只读检查与发布

在研究仓库执行：

```sh
python3 -B experiments/riichi-ranked-100-20261028-v1/audit_ranked_continuation.py
```

默认读取外置原首场、原 stopped batch 和新 continuation78。`--first-game`、`--stopped-campaign`、`--continuation` 可以指定对应证据的路径；模型身份、旧审计器摘要、历史文件摘要及本次恢复启动时间不可通过 CLI 改写。

| 状态 | 含义 |
| --- | --- |
| `pending` | 后续运行尚未完整结束或快照正在更新；没有最终验收文件 |
| `invalid` | 完整性、身份、记录一致性或预定恢复结构不符；没有最终验收文件 |
| `ready_with_interruption` | 独立核对了 101 次尝试、100 场完整及唯一固定中断；只读结果尚未发布 |
| `completed_with_interruption` | `--write` 已独占发布上述审计；原无故障门槛仍为 false |

只读命令的 pending/invalid 返回码为 1，ready 返回码为 0。所有状态都保留 `original_zero_failure_criterion_met=false`。`final_acceptance=true` 的范围仅为本恢复方案的记录核对与完整场数达标，不能转述为无故障运行通过。

后续 78 场全部完整结束后：

```sh
python3 -B experiments/riichi-ranked-100-20261028-v1/audit_ranked_continuation.py --write
```

发布会重新独立审核全部源文件和身份，比较整个只读结果，再复核源文件字节和当前审计器摘要，最后以独占方式创建 `continued-audit/`。既有目录不覆盖；尚未就绪或写入拒绝返回码为 2。

| 文件 | 内容 |
| --- | --- |
| `report.json` | 全部审计、恢复证据、统计口径、身份及原始哈希索引 |
| `attempts.json` | 全部 101 次尝试；含总第 23 次中断，无成绩筛选 |
| `raw-sha256.json` | 410 个原始证据文件的绝对路径、大小及 SHA-256 |
| `SHA256SUMS` | 以上三份派生文件的完整性摘要 |

最终 report 与只读 fresh 结果只有三个差异：status 改为 `completed_with_interruption`、final_acceptance 改为 true、增加 published_at。后续模型清单应重新调用本工具审计全部源证据并逐字段比较，而不能只相信已写报告的状态字段。

## 固定历史与完整性门禁

首场四文件、原 batch 三份父记录、失败子目录四文件共 11 项大小和 SHA-256 固定在新脚本中。原 batch 必须维持 stopped、requested=99、attempted=22、completed/healthy=21。失败子目录必须仍是唯一指定的 connection_closed/零事件/未知 seat 和终分，并与其父记录四个错误标记一致。即使另算摘要，也不能把该失败改为成功或声称它没有服务器结果。

完整场沿用并绑定旧审计器的固定 SHA-256，以其逐事件检查器核对完整 start_game/end_game、seat/终分、decision 与 accepted ack 的 request_id 一一对应、非零一致的请求/准备/发送/accepted 计数，以及零降级/超时/非法等原有门槛。原 batch 与 continuation 的逐场索引、计数及耗时汇总分别重算。任何新增失败、遗漏、额外目录或记录不符，都阻止本版本发布；不会补另一场来悄悄替换失败尝试。

`integrity_errors` 表示证据审核问题，`attempt_failures` 保留预先固定的历史中断；两者分别展示。`attempts.json` 使用总尝试序号和完整场序号：历史失败为 attempt 23、completed_game=null；续跑首场为 attempt 24、completed_game=23，最终为 attempt 101、completed_game=100。

统计以已核验完整场为分母；中断单独展示，不虚构得分、顺位或评级。所有完整场的动作合并后计算线性插值 p95；旧 supervisor 的 nearest-rank p95 只用于核对历史索引。本地 decision 延迟与服务器 elapsed 分开统计。终分只减 25,000 得出得失点；同分保留顺位区间，不猜服务器同分规则。在线匹配结果是描述统计，不替代固定对手棋力评估。

## 测试与调用接口

合成测试只在系统临时目录创建证据，不接触运行中的数据：

```sh
python3 -B -m unittest discover \
  -s experiments/riichi-ranked-100-20261028-v1 \
  -p test_audit_ranked_continuation.py -v
```

Python 接口：`audit_ranked_continuation(first_game, stopped_campaign, continuation) -> dict`。生产调用只传三条路径。`paths`、`expected`、`history_pins` 覆写仅供合成测试，正式发布不得使用。
