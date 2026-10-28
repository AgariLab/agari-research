# 既有训练曲线浏览器验收

2026-10-28，在独立 `fix/tensorboard-viewer` worktree 和外置工具环境中，实际启动
TensorBoard 2.20.0，用本机 Chrome 打开 Scalars，确认既有 GRP、CQL、PPO 曲线已绘制。
本次没有重训，没有读取 held-out 对局，也没有改变模型、检查点或原事件文件。
验证时该分支尚未合入主线；运行中排位所用的主锁文件和主 `.venv` 保持冻结。

首次直接使用主研究 `.venv/bin/python -m tensorboard.main` 启动失败：
TensorBoard 2.20.0 的 `default.py` 导入 `pkg_resources`，但已锁定的 setuptools 84.2.1
不提供该模块。[原始错误](initial-failure.log)和[失败清理记录](initial-failure.json)均保留。
项目原有 `build-system` 中的 `<82` 约束只作用于构建环境；本分支将
`setuptools>=77.0.3,<82` 同时加入运行时依赖，离线锁定到 81.0.0。
逐包比较确认其他依赖版本未变，清华索引未变。未使用临时 `PYTHONPATH` 替代项目修复。

在新外置 `runs/tensorboard-ui-v1/project-env` 执行 `uv sync --frozen --offline` 成功，
共安装 43 包；独立环境包含正常研究依赖，原环境 setuptools 仍为 84.2.1。
新增 `tests/test_tensorboard.py` 写入一条临时标量后运行真实 CLI `--inspect`：1 项通过，
13.254 秒。`uv lock --check --offline` 通过。该测试会被现有 CI 的 `tests` discover 运行，
负责捕获 CLI 导入与事件解析故障；下述浏览器记录负责实际绘制验收。

最终截图全部使用 **Smoothing = 0**，关闭 **Ignore outliers in chart scaling**，横轴为 step。
TensorBoard 的默认异常值裁剪会让只有四点的 PPO 图缺点，因此先关闭该设置再验收。
搜索框产生的匹配区位于页面顶部，原分类仍显示在其下方。

| 阶段 | 实际筛选 | SVG 绘图点数与终步 | 截图 |
|---|---|---|---|
| GRP | `^grp/(loss\|lr\|complete_epochs)$` | 每条抽样 1,000 点，终步 1,133；complete_epochs 末值 1 | [GRP](grp-final.png) |
| CQL | `^cql/(loss\|conservative_loss\|td_loss)$` | 每条抽样 1,000 点，终步 8,217 | [CQL](cql-final.png) |
| PPO | `^ppo/` | 8 张图，每条完整 4 点，终步 4 | [PPO](ppo-final.png) |

这些点数来自浏览器中实际 SVG path 绑定的数据；同时记录路径长度、非零绘制范围与有限值检查。
GRP/CQL 的界面默认抽样点数不等于原始更新总数，CQL 此次抽样的首点是 step 6。
三张截图均已人工打开核对。本记录证明训练曲线查看能力，不据此宣称棋力提高。

事件原件为外置
`runs/full-coverage-trace-1000-v1/tensorboard/events.out.tfevents.1793131978.zhou-mac-mini.9260.0`，
5,408,636 字节；查看前后 SHA-256 均为
`8ff3a23221e8bab6ef7f50ab410572bbe070be2dca970396604e6abe91c6707c`。
主研究 `uv.lock` 前后均为
`e77f024ebd5a8f4ab8cc686ce3ec3e6acccb7a997f574d02f3bf0679bd4e0406`。

实际服务绑定 `127.0.0.1:49725`；Chrome 使用独立临时 profile，目标请求仅允许此本地源，
52 次 HTTP 请求均在 localhost，外站请求和 JavaScript 异常均为 0。
结束时只停止本次独立进程组：TensorBoard PID 20144、Chrome PID 20712 均退出 0，
PID 已消失，端口已关闭，无该 profile 的残留进程，临时 profile 已删除。
最初失败的 TensorBoard PID 96199 也已退出并完成清理。

[verification.json](verification.json)汇总最终检查；[evidence.json](evidence.json)保留原始浏览器操作、
首次默认异常值裁剪的探索视图、最终视图、请求记录和清理结果。探索视图由最终视图替代，
仓库只附最终三图。原始小型证据和采集脚本位于外置 `runs/tensorboard-ui-v1/`，
这里不复制事件文件、完整环境、缓存或牌谱。文件校验见 [SHA256SUMS](SHA256SUMS)。

在本次独立 worktree 重现查看入口（端口可替换为空闲端口）：

```sh
cd /Users/zhou/MyCodes/AgariLab/agari-research-tensorboard
UV_PROJECT_ENVIRONMENT='/Volumes/新加卷/AgariLab/runs/tensorboard-ui-v1/project-env' \
  python3 ../agari-kit/scripts/with_storage.py uv run --frozen --offline tensorboard \
  --logdir '/Volumes/新加卷/AgariLab/runs/full-coverage-trace-1000-v1/tensorboard' \
  --host 127.0.0.1 --port 6006 --load_fast=false --reload_interval 0
```

打开 `http://127.0.0.1:6006/#scalars`，调整上述显示选项并输入筛选表达式；结束按 `Ctrl-C`。
这是只读查看既有日志的命令。实际验收使用同一环境的 `python -m tensorboard.main`，
完整参数和 Chrome 启动命令在原始机器证据中。
