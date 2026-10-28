# 独立排位验收工具

`audit_ranked.py` 只读首场和 remaining99 的本地记录，不导入运行中的 agari-kit，不加载模型，不读取凭证或环境值，也没有网络调用。固定模型、descriptor、kit/research 源码、原生引擎和 supervisor 摘要均对实际文件复算。因此应在本批固定实现仍可核对时完成最终审核；源码改变会阻止通过。

从研究仓库执行只读探测：

```sh
python3 -B experiments/riichi-ranked-100-20261028-v1/audit_ranked.py
```

默认源目录为外置 `runs/riichi-ranked-first-v1/` 和 `runs/riichi-ranked-campaign-remaining99-v1/`。可用 `--first-game`、`--campaign` 指定对应完整记录目录；生产身份摘要不可由 CLI 改写。

- `pending`：尚未完整结束，或运行中快照正在变化；不产生任何最终报告。只审查已登记且 manifest 完整的场次，不读取进行中场次的 events。运行中的 JSONL 尾行或父记录更新时序不作为完工证据。
- `invalid`：发现失败、缺失、身份变化、数据不一致或无法独立核验；不产生最终报告。
- `ready`：只读独立审核了全部 100 场且通过；仍然 `final_acceptance=false`，尚未发布验收。

默认命令返回码：ready 为 0，pending/invalid 为 1。显示的进度是该次只读快照，不能当作最终 100 场结论。

确认全部结束后执行：

```sh
python3 -B experiments/riichi-ranked-100-20261028-v1/audit_ranked.py --write
```

`--write` 要求重新独立审核仍为 ready，随后再次核对全部原始文件字节，才独占创建 `final-audit/`。既有目录拒绝覆盖，拒绝发布时返回码为 2。输出仅有四个小型派生文件：

| 文件 | 内容 |
| --- | --- |
| `report.json` | 独立审计、身份核验、统计方法、限制及原始哈希索引 |
| `games.json` | 全部 100 场逐场派生指标 |
| `raw-sha256.json` | 首场、99 个子目录和 campaign 三份索引文件的路径、大小及 SHA-256 |
| `SHA256SUMS` | 以上三个文件的完整性摘要 |

发布报告仅在只读结果基础上把 `status` 改为 `passed`、`final_acceptance` 改为 true，并增加 `published_at`。其余核心字段确定不变，供后续模型清单重新审核和逐字段比较。工具本身摘要也记入报告。不会复制观察数据、模型权重或凭证。

审核要求每场持有唯一、完整的 start_game/end_game 记录，seat 和终分与 game/report/campaign 索引相符；request、准备/发送回复、accepted 计数均非零相等，并逐 request_id 匹配 decision 和 accepted ack。超时、拒绝、非法、默认代打、stale ack、未知 ack 和降级均阻止验收。所有失败尝试保留在索引与审计行中，不因成绩筛选。缺场、额外场、乱序索引、重复整场事件证据和不一致的父 summary 都不能发布。

非致命的 duplicate_start、duplicate_requests、stale_requests、duplicate_acks 和 unknown_messages 与既有 supervisor 门槛一致，单独核对和汇总，不新增零值要求。原客户端会过滤这些消息而不写事件，因此报告明确这些计数只能从保留的计数记录交叉核对，无法从已省略的消息重建。

动作耗时由所有场次的全部动作合并后计算线性插值 p95，不平均各场 p95。supervisor 原有 nearest-rank p95 仅被复算以核对其记录。本地 decision 计收包到准备回复之间的处理时间；服务器 elapsed 包含请求/回复网络往返，两者分开报告。终分只减初始 25,000 计算得失点；不混入评级、uma 或 oka。分数相同时保留 rank_min/rank_max 区间，不猜服务器同分顺位规则。

本工具审核的是保留的传输记录，不是完整 MJAI 重放或服务器签名证明；100 场在线匹配结果只作运行可靠性和描述统计，不替代固定对手棋力筛选。

合成门禁测试仅在系统临时目录构建数据：

```sh
python3 -B -m unittest discover \
  -s experiments/riichi-ranked-100-20261028-v1 \
  -p test_audit_ranked.py -v
```

Python 调用接口为 `audit_ranked(first_game, campaign) -> dict`。`paths` 和 `expected` 关键字覆写只用于合成测试，正式审核必须使用默认固定身份。
