# 独立 seeded trace 选中动作计数

固定原版 RiichiEnv 0.4.10 二进制；从原始 trace 重放并匹配完整 Action.to_dict，在执行同时响应优先级前统计实际提交动作。仅 len(obs.legal_actions()) > 1 计策略样本。未使用 MjaiReplay.steps 或新 trace_samples，未读取 test 内容。

| 数据集 / 分区 | 场数 | 局数 | GRP样本 | 策略样本 | 旧native策略样本 | 差值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| meld-coverage-v1 / train | 90 | 835 | 3340 | 56429 | 56419 | +10 |
| meld-coverage-v1 / validation | 5 | 46 | 184 | 3196 | 3195 | +1 |
| selfplay-v1 / train | 900 | 9057 | 36228 | 525853 | 525721 | +132 |
| selfplay-v1 / validation | 50 | 420 | 1680 | 24603 | 24603 | +0 |

| 实际选中动作 | meld train | meld validation | selfplay train | selfplay validation |
| --- | ---: | ---: | ---: | ---: |
| ANKAN | 3 | 0 | 0 | 0 |
| CHI | 1410 | 79 | 0 | 0 |
| DAIMINKAN | 11 | 1 | 0 | 0 |
| DISCARD | 43971 | 2483 | 384368 | 18155 |
| KAKAN | 19 | 0 | 0 | 0 |
| KYUSHU_KYUHAI | 2 | 0 | 0 | 0 |
| PASS | 8784 | 513 | 117951 | 5397 |
| PON | 655 | 32 | 0 | 0 |
| RIICHI | 952 | 54 | 15395 | 683 |
| RON | 446 | 25 | 6096 | 310 |
| TSUMO | 176 | 9 | 2043 | 58 |

所有分区 GRP 数均与旧审计相同。逐场验证新增样本数恰好等于额外多家 RON 玩家数加九种九牌动作数；这条等式只解释样本数量，不代替旧/新标签逐样本对齐。

meld train 动作分布差为 CHI +27、PON +7、PASS −34、RON +8、KYUSHU_KYUHAI +2；validation 为 CHI +2、PON +1、PASS −3、RON +1。这里只读既有 native 审计计数，不调用旧 reader 重新生成标签。

每场 JSONL 包含原始 trace/MJAI/receipt 文件hash、规范MJAI hash、seed、全部/强制/可选动作分布及多家RON step。两个数据集 JSON 绑定冻结清单摘要和旧报告hash；report.json 绑定审计脚本、引擎和数据集报告hash。
