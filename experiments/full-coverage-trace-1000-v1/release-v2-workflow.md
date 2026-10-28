# 100 场排位后的模型发布清单 v2

`build_model_release_v2.py` 只验证已有工件，不训练、不连接牌局、不读取凭证。
当前排位批次尚未通过最终独立验收，不能创建或手写 `release-v2.json`。
默认执行为只读：未完成时输出 `pending`，不生成发布清单。

v2 固定继承两份字节相同的 v1，SHA-256 为
`8e5d2467cda1a4f0570f64eceb3a862e253d6bfcd81e235cd31133c1d15a4f1a`。
它先核对原 v1 生成器的摘要并重新执行其模型、训练、离线评测和真实验证证据检查，
再直接调用独立 `audit_ranked.py` 重新检查排位原始文件。
单独存在一个标记为 `passed` 的报告不足以通过：最终报告除发布状态与时间外，
必须与本次独立审核结果逐字段相同，逐场记录、原始文件索引及校验文件也必须一致。

所有命令从 `agari-research` 执行，使用既有锁定环境。
**必须在合并任何 kit／research 运行源码变更之前生成并验证 v2。**
检查器要求实际源码、引擎和权重仍匹配记录；日后复核需要恢复相应固定运行环境。
它不会把当前 Git HEAD 回填为历史执行版本。

先执行只读检查；这一步可在批次运行时使用：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/full-coverage-trace-1000-v1/build_model_release_v2.py
```

只有独立审核确认首场加后续批次的完整 100 场、原始请求与回执、结果及来源身份均通过，
才依次执行以下命令。最终审核与 v2 发布均拒绝覆盖既有文件，不自动删除失败证据。

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/riichi-ranked-100-20261028-v1/audit_ranked.py --write
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/full-coverage-trace-1000-v1/build_model_release_v2.py --write
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/full-coverage-trace-1000-v1/build_model_release_v2.py --verify-release
```

最终审核单独写入 `experiments/riichi-ranked-100-20261028-v1/final-audit/` 的
`report.json`、`games.json`、`raw-sha256.json` 和 `SHA256SUMS`。
v2 随后新建两个字节相同的文件：

- 外置 `models/full-coverage-trace-1000-v1/release-v2.json`。
- 仓库 `experiments/full-coverage-trace-1000-v1/model-release-v2.json`。

v1、权重、模型描述和所有历史报告均不修改。外置卷不支持硬链接，写入采用独占创建；
不会覆盖并发创建的文件。进程写入失败会清理本次不完整文件，异常断电仍可能留下
不完整新文件；应人工检查，不能用覆盖旧版的方式“修复”发布。
两个副本不构成跨文件系统事务：若外置副本已写成功、仓库副本失败，程序返回
`status=partial` 和 `created_paths`，保留已完成副本，不谎报完全未写入。
此时应检查已创建文件及失败原因；不能直接重复 `--write` 来覆盖或重新发布。

训练提交 `761b452`、离线推理提交 `723b810`／包摘要 `44bb7f08…`、真实验证及
本次排位的研究包摘要 `43eabb39…` 分别保留。验证局为东风；训练、本地评测和
排位为四人赤牌半庄。100 场仅满足首期线上工程样本要求，不宣称正式晋级或榜单目标。

默认 `pending` 退出码为 0；携带 `--write`／`--verify-release` 时遇到未完成条件为 2；
工件不一致、身份不匹配或无效审核为 1。仅成功写入后才输出 `status=published`。
发布门禁的独立合成测试只使用临时目录，不会创建真实 v2：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/full-coverage-trace-1000-v1/test_model_release_v2.py -v
```
