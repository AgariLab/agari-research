# 中断后续跑的 v2 发布流程

> 历史固定方案，当前不适用：continuation78 已在完成 6 场后再次中断。
> 本工具仍保留原门槛并拒绝发布，不得用于生成真实最终审核或 release-v2。
> 当前全部尝试和后续诊断进度见[排位记录](../riichi-ranked-100-20261028-v1/README.md)。

本流程对应新的 `riichi-ranked-continuation78-v1`，使用独立
`build_model_release_v2_continued.py`。原 v1、原 `build_model_release_v2.py`、
原严格排位审核器和原流程文档均保留；原生成器继续拒绝未完成的 99 场批次。
尚未发布的 v2 只能在以下续跑条件实际全部成立后生成。

| 来源 | 计划场数 | 已要求保留的尝试 | 完整对局 | 失败尝试 |
| --- | ---: | ---: | ---: | ---: |
| `riichi-ranked-first-v1` | 1 | 1 | 1 | 0 |
| `riichi-ranked-campaign-remaining99-v1` | 99 | 22 | 21 | 1 |
| `riichi-ranked-continuation78-v1` | 78 | 最终 78 | 最终 78 | 必须为 0 |
| 最终合计 | — | 101 | 100 | 1 |

这里的 100 场是 `22 + 78` 个完整对局。旧 99 场批次仍为 `stopped`，
并未完成，也不被改写为一个新的 21 场成功批次。
全局第 23 次尝试（旧批次 `game-000022`）记录 `connection_closed`：
本地 `events.jsonl` 为零字节，没有观察到 `start_game` 或动作请求，席位与结果均为空。
这只能说明本地没有相关事件；**服务端是否开局、对局结果和评分影响仍未知**。
不能用“未开局”或“无评分影响”填补证据。

最终 v2 必须明确记录：

- `status=completed_with_interruption`。
- `completed_games=100`、`attempts=101`、`failed_attempts=1`。
- `original_zero_failure_criterion_met=false`。
- `original_99_game_batch_completed=false`。
- 原始 101 次尝试和 410 个源文件的完整独立审核索引，包含失败尝试。

“达到完整对局数量”与“原定零失败条件”是两项不同结论。恢复后不能声称 100 次连续成功，
也不能把未知结果分配一个名次或得分。新续跑再发生一次中断或任何审核完整性错误，
本流程将拒绝发布；不能自行过滤失败、扩大目标或换一份成功清单。

所有命令从 `agari-research` 执行，使用既有锁定解释器，不修改环境。
**发布和发布后复核必须在合并 kit／research 运行源码变更之前完成。**
v1 的训练、1,000 场离线评测及真实验证工件会重新核验；原 v2 helper 仅用于固定摘要的
v1 校验和独占写入，不调用它已经不适用的原排位完成条件。

运行中的只读检查：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/full-coverage-trace-1000-v1/build_model_release_v2_continued.py
```

默认未完成时输出 `pending`、`release_written=false`，不会生成伪最终清单。
100 个完整对局实际齐备后，按以下顺序发布独立审核、生成 v2、再复核 v2：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/riichi-ranked-100-20261028-v1/audit_ranked_continuation.py --write
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/full-coverage-trace-1000-v1/build_model_release_v2_continued.py --write
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/full-coverage-trace-1000-v1/build_model_release_v2_continued.py --verify-release
```

新的独立审核只发布 `experiments/riichi-ranked-100-20261028-v1/continued-audit/` 中的
`report.json`、`attempts.json`、`raw-sha256.json` 和 `SHA256SUMS`，不覆盖 `final-audit/`。
生成器重新调用审核器，以原始文件复算后逐字段比较已发布审核；单独的成功标签不足以通过。
除摘要外，它也直接复查旧父清单／摘要、失败四文件以及新批次的 78 场完成状态。

成功后新建两份字节相同的清单：

- 外置 `models/full-coverage-trace-1000-v1/release-v2.json`。
- 仓库 `experiments/full-coverage-trace-1000-v1/model-release-v2.json`。

两处都必须不存在。外置写入使用固定旧 helper 的独占创建，不依赖硬链接，也不覆盖已有文件。
两个副本不是跨文件系统事务；若第二处失败，程序报告 `partial` 和已创建的路径，保留第一份。
必须检查原因与已有副本，不能盲目重复 `--write`。机器异常断电也可能留下不完整新文件。

默认 `pending` 退出码为 0，要求写入／验证却尚未完成为 2，证据错误或部分写入为 1。
`status=published` 仅表示两个清单均写入；其中业务状态仍是 `completed_with_interruption`。
训练源码 `761b452`、离线推理 `723b810`／包 `44bb7f08…`、真实验证及排位包 `43eabb39…`
分别保留，不回填当前 Git HEAD。正式棋力晋级和榜单里程碑仍不在本版范围内。

合成门禁测试只使用临时目录，不产生真实审核或发布清单：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/full-coverage-trace-1000-v1/test_model_release_v2_continued.py -v
```
