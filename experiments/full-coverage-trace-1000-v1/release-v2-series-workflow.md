# 保留全部排位尝试的 release v2

`build_model_release_v2_series.py` 为完整 series 新增独立发布门禁，不修改旧 v1、原严格 v2
生成器或一次中断续跑生成器。新清单要求恰好 100 个健康完整排位，同时保留每次中断及其未知
服务器结果。完成目标和数据完整性单独核对，原连续零故障标准始终为 false。

当前真实 series 尚未完成；本流程及测试不代表已发布独立审计或 release。默认运行只读，
未完成时返回 pending，不创建任何真实发布文件。

## 固定门禁

1. 先核对原 `build_model_release_v2.py` SHA256
   `5368fbf3b2f14288b52af52df0ccef67caa11f5b6a1fb8a2d8cc9f937bc16ee8`，复用其 `verify_v1`。
   两份 v1、原 v1 生成器及全部训练、评估、验证、模型输入必须重新通过验证。
2. 调用独立 `audit_ranked_series.py` 的生产接口，不传入测试身份。必须 ready、integrity verified、
   没有待处理项或完整性错误；恰好 100 个完整且健康的局，不截取超过目标的结果。
3. 历史未加观察器的运行固定为 30 次尝试、28 个完整局、2 次中断；后续观察器组须贡献 72 个完整局。
   两个旧中断分别保留 global attempt 23/30、0/115 次请求及未知服务器结果和评级影响。
   新增中断也须全部保留；不能用新完成局替换或隐藏失败尝试。
4. 模型、descriptor、推理包、工具包、引擎、supervisor 必须匹配不可变 v1。
   `parent_transport_observer` 与 `uninstrumented` 分组保留；观察器 SHA 固定
   `9c8dd327662276771082a5dd543b7bd4b00d8fa419ce3de64962a808cd10e294`。
5. 独立 `series-audit/` 必须先发布，恰含 `report.json`、`attempts.json`、`source-sha256.json`、
   `SHA256SUMS` 四文件。校验和正确后，仅规范化发布时间和发布状态，再与 fresh 审计逐字段、
   逐 JSON 类型比较；成功标签不能替代实际核验。
6. source inventory 中每份源文件重新核对 bytes/SHA，原始日志须为其子集，所选 registry head、
   全部祖先和新旧执行计划均须保留。发布准备结束前再次运行独立审计及 v1 验证，结果必须不变。

series 根 registry 固定为
`564d5f67baaa9dd0c901313ae3c5879b4e81ffb437c1d7e38c5f760a4fa5cc2f`。
默认读取 `series-registry-v1.json`；显式 `--registry` 可以指定已登记的追加链新 head。
当前生产 head 为 `series-registry-v3.json`，SHA
`6f3f279caafcce2dabb8dc3b3747f0757eb4e33d2755c81626f297702de4ce5d`；
以下命令均显式选择 v3，保留全部四次中断及新登记的 10 个尝试。
独立审计验证 predecessor 摘要、连续 revision、旧计划前缀及不可变历史，不能借换 head 换 root。
审计发布与 release 准备必须使用同一个 head；新 head 不允许覆盖已有独立审计或 release。

## 执行与发布顺序

在保留已记录源码、模型和引擎的研究仓库锁定环境中，先只读检查：

```sh
.venv/bin/python -B experiments/full-coverage-trace-1000-v1/build_model_release_v2_series.py \
  --registry experiments/riichi-ranked-100-20261028-v1/series-registry-v3.json
```

真实 100 个健康完整局齐备、全部实际尝试已结束并经审查后，依次执行：

```sh
.venv/bin/python -B experiments/riichi-ranked-100-20261028-v1/audit_ranked_series.py \
  --registry experiments/riichi-ranked-100-20261028-v1/series-registry-v3.json --write
.venv/bin/python -B experiments/full-coverage-trace-1000-v1/build_model_release_v2_series.py \
  --registry experiments/riichi-ranked-100-20261028-v1/series-registry-v3.json --write
.venv/bin/python -B experiments/full-coverage-trace-1000-v1/build_model_release_v2_series.py \
  --registry experiments/riichi-ranked-100-20261028-v1/series-registry-v3.json --verify-release
```

若采用后续 registry head，为上述独立审计和三种发布器模式显式传入同一个
`--registry experiments/riichi-ranked-100-20261028-v1/series-registry-vN.json`；`vN` 是实际已登记
的新文件名，不自动创建或递增 registry。原严格 `final-audit/` 与一次中断 `continued-audit/`
均保持独立，不被 series 审计覆盖。

成功时只新建两份字节相同的清单：

- 外置 `models/full-coverage-trace-1000-v1/release-v2.json`。
- 仓库 `experiments/full-coverage-trace-1000-v1/model-release-v2.json`。

两处任何一处已存在（包括符号链接）都拒绝写入。创建复用冻结旧 helper 的独占写入，
不依赖跨卷硬链接。双副本不是事务：第二处失败时返回 partial，保留第一份并报告已创建路径。
不得盲目重试或删除已有结果；应先检查原因与原件。断电也可能留下需人工核对的不完整新文件。

默认 pending 的退出码为 0；请求写入或验证但仍 pending 为 2；证据错误或部分发布为 1。
成功写入的操作状态为 published，但清单业务状态为 `completed_with_interruptions`，
`original_zero_failure_criterion_met=false`。发布清单保留整个 fresh series 报告，包括完整尝试、
中断、未知结果、批次、计数范围、观察器分组及审计新增字段，不压缩成只含成功局的摘要。

v1 全部字段仍被保留：未改变的证据沿用原位置，被新版本元数据替换的原字段存于
`inherited_v1_fields`。训练、离线评估、线上验证的历史来源提交不被当前 Git HEAD 回填。
这仍不代表正式棋力晋级或 1,000 场/七天榜单里程碑。

## 本地门禁测试

测试只使用临时合成文件、模拟独立审计结果及冻结 helper 的独占发布函数，不连接平台，
不读取凭证，也不创建真实模型发布文件：

```sh
.venv/bin/python -B experiments/full-coverage-trace-1000-v1/test_model_release_v2_series.py -v
```

覆盖历史信息保留、100 局精确门槛、旧/新中断、未知结果、runtime 分组、已发布审计全字段匹配、
源码清单漂移、追加 registry head、祖先与旧计划遗漏、pending 不写入、旧目标保护及双副本部分失败。
