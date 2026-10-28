# 历史 release 的运行环境与复核

release-v2 已在冻结环境发布，并两次通过 `--verify-release`。两份清单均为
887,451 字节，SHA-256 为
`8edf5751901cadfa02af084a12e5d2048894c28a5e704d2778894c5fc1a09e2e`。
它保留 104 次实际尝试中的 100 个健康完整局和 4 次中断；状态为
`completed_with_interruptions`，原零故障标准仍为 false。

在合入 replay、TensorBoard 修复及更新主环境之前，旧 research 环境和真实 CPython
目录已保存到 `/Volumes/新加卷/AgariLab/runs/release-runtime-20261028-v1`。
完成时间为 `2026-10-28T15:03:40.155707+00:00`。实际 `manifest.json` 为 8,864 字节，
状态 `complete`，SHA-256 为
`6ab7a20882191360c9736d0b84b6378154e3d3b5a08fcb8e72445ca1056f7a64`。
精简机器回执见 [runtime-preservation.json](runtime-preservation.json)；完整清单和逐成员清单保留在外置备份目录。

| 归档 | 字节数 | 成员数 |
| --- | ---: | ---: |
| `research-venv.tar` | 730,869,760 | 27,762 |
| `cpython-3.12.16-macos-aarch64-none.tar` | 75,028,480 | 2,107 |

两份无压缩 PAX tar 保留符号链接，不读取链接指向的外部内容。保存脚本逐成员检查了路径、
类型、权限模式、普通文件大小与 SHA-256、符号链接目标，并在完成前重新核对源树未变。
完成清单记录 `source_writes=false`、`restoration_performed=false`。
这证明备份已完成；**尚未实际恢复，也未完成跨机迁移验收**。

今后复核应使用以下冻结版本，而不是更新后的 main：

| 仓库 | 冻结 commit | 必须呈现的 canonical 路径 |
| --- | --- | --- |
| agari-research | `abc2f3b61a829d9bdfd0e06f0bd0f9b37ce17e39` | `/Users/zhou/MyCodes/AgariLab/agari-research` |
| agari-kit | `44003b3dd5f9cd993a9a1226ba08bf905df0edf1` | `/Users/zhou/MyCodes/AgariLab/agari-kit` |

备份时两仓库均无已跟踪文件改动。research 的冻结提交包含已发布的两版仓库清单、
series-audit、registry 链和原验证器；其 Git 历史还须保留验证器读取的训练提交
`761b4522b5343b01413d640b58444b8a3fb7568e` 与评估提交
`723b81090944ecc430c2be331ba1cba364ab41b6`。

旧环境使用 CPython 3.12.16、Torch 2.9.1、NumPy 2.4.1、RiichiEnv 0.4.10。
历史安装版本还包括 TensorBoard 2.20.0 和 setuptools 84.2.1；这是备份身份，
不应据此撤销当前主环境的 TensorBoard 修复。旧 research `uv.lock` 的 SHA-256 是
`e77f024ebd5a8f4ab8cc686ce3ec3e6acccb7a997f574d02f3bf0679bd4e0406`。

research `.venv/bin/python` 通过 uv 别名引用真实解释器，恢复 `.venv` 本身还不够：

- 真实目录：`/Users/zhou/.local/share/uv/python/cpython-3.12.16-macos-aarch64-none`。
- 别名：`/Users/zhou/.local/share/uv/python/cpython-3.12-macos-aarch64-none`，指向上述真实目录。
- `.venv/bin/python` 链接目标：上述别名下的 `bin/python3.12`。
- `pyvenv.cfg` 的 `home`：上述别名下的 `bin`；原配置文本及解释器、引擎指纹见精简回执。

完整 v2 复核会按路径读取原 kit/research 源树、research `.venv` 中的引擎、
`/Volumes/新加卷/AgariLab` 下的模型和运行原件，并逐字段比较已发布审计中的绝对路径。
独立 worktree 加 `PYTHONPATH` 不能满足这些约束；只改变导入路径也不能替代原路径下的文件。
外置存储必须在原路径真实挂载，不能用任意路径重定向来绕过原件检查。
复核应安排在维护窗口，或在兼容 macOS/arm64 的隔离环境中呈现这些 canonical 路径。
本说明不执行恢复，也不要求在日常开发目录直接覆盖当前环境。

使用的登记 head 必须仍为
`experiments/riichi-ranked-100-20261028-v1/series-registry-v3.json`，SHA-256 为
`6f3f279caafcce2dabb8dc3b3747f0757eb4e33d2755c81626f297702de4ce5d`。
保留 v1、v2 祖先及三份执行计划；不能修改清单内的路径或改用默认 v1 head。

旧 auditor 会扫描 `/Volumes/新加卷/AgariLab/runs/riichi-ranked-*` 并拒绝未登记目录。
发布时共有 77 个登记的运行根目录，完整名称保存在外置完成清单的
`identity.ranked_directories_at_publication`，精简回执只记录数量。
后续排位实验须与这份冻结证据视图隔离；未来复核时应呈现发布时的目录集合。
不能删除、改写或移动当前实验日志来伪造旧集合。旧计划未启动的 38 个尾部目录也应保持未执行。

后续复核的最小流程如下：

1. 先用精简回执中的 SHA-256 核对外置完成清单，再按完整清单核对归档和成员清单。
   缺少 `manifest.json`、状态不是 `complete`，或校验不符时，应将备份视作未确认，不能盲目重跑保存脚本。
2. 在单独安排的恢复环境中准备上述冻结仓库、旧锁文件、research 环境、真实解释器及别名，
   并提供原模型、训练/评估/验证证据、全部登记排位原件和已发布审计。恢复后再核对源文件与环境指纹。
3. **只有恢复了冻结版本及原路径之后**，在该环境运行下面两个只读命令。
   当前 main 已更新源码和环境，不能直接在当前 main 运行它们来检验旧发布是否失效。

```sh
cd /Users/zhou/MyCodes/AgariLab/agari-research
AGARILAB_STORAGE_ROOT='/Volumes/新加卷/AgariLab' .venv/bin/python -B \
  experiments/full-coverage-trace-1000-v1/build_model_release.py --verify-release
AGARILAB_STORAGE_ROOT='/Volumes/新加卷/AgariLab' .venv/bin/python -B \
  experiments/full-coverage-trace-1000-v1/build_model_release_v2_series.py \
  --registry experiments/riichi-ranked-100-20261028-v1/series-registry-v3.json \
  --verify-release
```

这些命令读取既有证据并本地检查模型张量；不会启动 Bot、重新进行平台 validation、训练或发布。
不要加 `--write`。旧验证器对当前源码、锁文件或环境漂移的拒绝属于预期保护，不能通过修改旧清单规避。

运行时备份另含 v2 清单与独立审计副本，**不包含全部 datasets、models、checkpoints、运行日志或 Git 仓库历史**。
必须继续保留外置原件和冻结 Git 历史；不能把两个环境 tar 当作整个实验的可恢复备份。
归档验证与实际恢复验收是两个不同结果，当前仅完成前者。
