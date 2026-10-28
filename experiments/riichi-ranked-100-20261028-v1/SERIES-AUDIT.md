# 固定目标、追加登记的排位系列审计

`audit_ranked_series.py` 将证据完整性、完整场数目标和运行故障分开判断。完整场目标固定为 100，不以增加或隐藏中断来重写通过标准。每一次实际尝试都保留；原有两次中断不会因后续达到 100 场而变成健康场次，`original_zero_failure_criterion_met` 仍为 false。

工具仅访问本机证据、源码和固定身份文件，不连接游戏服务器、不读取凭证或环境值，也不启动 bot。模型、客户端核心源码、本地引擎及监督脚本仍按实际字节核验；诊断 wrapper 另有固定身份。

当前生产 head 为 [series-registry-v3.json](series-registry-v3.json)，SHA
`6f3f279caafcce2dabb8dc3b3747f0757eb4e33d2755c81626f297702de4ce5d`。
第一份计划在第 42 次尝试连接打开超时后停止；第二份计划在第 21 次尝试握手成功后无关闭帧断开，
累计 90 完整、4 中断；两份旧计划的停止后缀均保持未执行。
v3 追加 [10 次新尝试计划](observed-series-plan-v3.json)，SHA
`95b74f69d815b320b571985282a692dd2ced5c1be7f0984baa895964a80c041b`，
不改旧 registry 或旧计划。以下当前生产命令显式选择 v3；省略参数仍会读取旧 v1。

## 初版登记

[series-registry-v1.json](series-registry-v1.json) 的根摘要为 `564d5f67baaa9dd0c901313ae3c5879b4e81ffb437c1d7e38c5f760a4fa5cc2f`，绑定四个既有 run 的全部 136 份原始文件：

| 批次 | 实际尝试 | 完整场 | 中断 | 运行方式 |
| --- | ---: | ---: | ---: | --- |
| first | 1 | 1 | 0 | 原始客户端 |
| remaining99 | 22 | 21 | 1 | 原监督脚本 |
| continuation78 | 7 | 6 | 1 | 原监督脚本 |
| diagnostic1 | 1 | 1 | 0 | parent_transport_observer |

初版基线为 31 次尝试、29 场完整、2 次中断；完整场请求 4,494，全部已核验终止尝试的观察请求 4,609。两者差 115，属于第二次中断，不能并入完整场成绩或完整场动作统计。

registry 同时绑定主代理在执行前写出的 [observed-series-plan-v1.json](observed-series-plan-v1.json)：最多 71 个独立单场 observer run，名称从 `riichi-ranked-observed-000001-v1` 到 `riichi-ranked-observed-000071-v1`。实际连接仍只由主代理执行。尚未创建的计划项保持 planned，不伪造为遗漏的实际尝试；已创建项必须按顺序出现，任一不成功 run 后禁止执行同计划后续项。

## 只读审核

从研究仓库运行：

```sh
python3 -B experiments/riichi-ranked-100-20261028-v1/audit_ranked_series.py \
  --registry experiments/riichi-ranked-100-20261028-v1/series-registry-v3.json
```

也可用 `--registry /absolute/path/to/registry-v2.json` 选择合法追加链的 head。工具必须递归验证到固定根摘要；旧历史、身份、目标和旧计划前缀都不得改写。

- `integrity_status=verified` 表示当前已核验终止记录自洽，不代表已达到 100 场。
- `status=pending` 表示目标尚未达到、存在运行中项或快照正在变化；只读检查不写最终验收。
- `status=invalid` 表示记录、身份、布局或执行顺序不符；实际尝试仍保留，不据成绩剔除。
- `status=ready` 要求恰好 100 场独立核验的健康完整场，所有实际源记录终止且稳定，没有完整性错误或待处理运行中项；此时 `final_acceptance` 仍为 false。

多于 100 场也不会被裁成最好的 100 场：全部记录保留，并阻止本版本发布。完整场数、健康完整数、中断数、损坏记录数和待定数分别给出。

工具会枚举所有已登记 run 及 `game-*` 子目录，并发现未登记的 `riichi-ranked-*` run。终止场固定只有 manifest/games/events/report 四文件；diagnostic 固定只有 manifest/events/result 三文件，额外文件、目录、空目录、符号链接均被拒绝。运行中项不读取当前 child 的 events；正常父索引追加、元数据替换及批次转换时序保持 pending，不冒充最终一致快照。

## 诊断运行方式与关闭阶段

`runtime_variants` 将 uninstrumented 和 parent_transport_observer 分组。observer wrapper SHA 固定为 `9c8dd327662276771082a5dd543b7bd4b00d8fa419ce3de64962a808cd10e294`，单场、无自动重连、代理策略 environment、ping interval/timeout 均为 20 秒，其余传输配置也逐项核对。核心源码和模型相同不代表运行时完全相同。

合法的握手前失败可能已有一次 connection_attempt，却尚未创建连接对象（connections_created=0）。审计根据事件和计数保留这种中断，不将其误当记录损坏；若已观察到 start_game/end_game，则必须有成功握手。

已核验的 end_game 和场次记录决定是否完整完成。receiver 取消、connection_lost、context_exit 或关闭阶段缺少 close frame 的诊断观察单独保存，不能把已经完整结束的健康场次追记成失败。诊断结果与场次结果矛盾、文件不完整或身份不符仍会阻止发布。

全部中断的服务器最终结果及评级变化保持 unknown。没有观察到 start_game 不证明服务器未开局；已进入对局后断线也不代表不计评级。本地引擎摘要只说明本机模块身份，不倒推服务器升级时间或断线根因。

## 统计与原始证据

`attempts` 按实际执行顺序保存全部尝试，`failures` 保留每次已核验中断及未知结果。完整场、中断场和全部已核验终止尝试分别汇总 counts 和动作耗时；运行中事件流没有被读入这些统计。

所有动作合并后采用 `(n - 1) × p` 线性插值计算分位数；原 supervisor 的 nearest-rank p95 仅用于核对父索引。本地 decision 与服务器 elapsed 分开统计。得失点只减初始 25,000；同分保留 rank_min/rank_max，不猜评级、uma、oka 或服务器同分规则。在线匹配结果不替代固定对手棋力评估。

`source_files` 覆盖全部已读原始记录、registry 链、每份计划、当前审计器、冻结旧检查器、诊断 wrapper，以及实际模型、descriptor、源码和原生引擎的路径、大小和 SHA-256。输出只保存这些摘要及小型派生记录，不复制权重或完整观察数据。

## 追加新计划

追加前需另写一份明确的新计划，沿用固定 target、模型和 observer 身份，使用全新的 run 名称。前一计划必须已 complete 或按首错停止为 stopped，且没有活动尝试；未执行的停止后缀继续保留。新计划基线必须等于之前实际完整/中断数，时间不得早于之前计划终止时间。

```sh
python3 -B experiments/riichi-ranked-100-20261028-v1/audit_ranked_series.py \
  --registry /absolute/path/to/registry-v1.json \
  --append-plan /absolute/path/to/new-plan.json \
  --output-registry /absolute/path/to/registry-v2.json
```

该命令只独占创建新 registry，加入 predecessor 路径及摘要和新的计划引用；不改旧文件、不启动任何连接。已创建的新 run 不能事后补登记为预计划。跨计划也核对终止、启动时间和禁止重叠规则。若需要改变模型、引擎或 wrapper，应另行明确登记运行身份及审核规则，不可借修改旧 registry 混称同一实现。

## 最终发布

达到 ready 后才可执行：

```sh
python3 -B experiments/riichi-ranked-100-20261028-v1/audit_ranked_series.py \
  --registry /absolute/path/to/selected-registry.json --write
```

发布前重新独立审核全部来源、比较完整结果，再逐份复查 source_files 的实际字节；以独占方式创建 `series-audit/`，已有目录不覆盖。四个文件为 report.json、attempts.json、source-sha256.json、SHA256SUMS。发布后的 report 仅将 status 从 ready 改为 published、final_acceptance 改为 true，并增加 published_at，其余字段必须与 fresh 审计完全相同。发布仍保留 original_zero_failure_criterion_met=false，不能转述成“100 场连续无故障”。

只读 ready 返回 0，pending/invalid 返回 1；被拒绝的 `--write` 返回 2。当前目标未完成时不运行真实发布。

Python 接口为 `audit_ranked_series(registry_path) -> dict`。生产仅传 registry 路径；其他 keyword 覆写只用于合成测试。

```sh
python3 -B -m unittest discover \
  -s experiments/riichi-ranked-100-20261028-v1 \
  -p test_audit_ranked_series.py -v
```

测试只在系统临时目录构造数据，覆盖两个历史失败保留、created=0 握手失败、关闭阶段诊断、完整性篡改、精确文件布局、时间戳、当前场禁读、同/跨计划时序、追加链和独占发布。
