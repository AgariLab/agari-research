# Akochan mixed-pool paired screening

Each model: 16 full hanchan, 4 independent seed groups (600500–600503). Difference = candidate minus baseline.

| Metric | Baseline [exploratory 95% interval] | Candidate [exploratory 95% interval] | Difference [paired exploratory 95% interval] |
| --- | --- | --- | --- |
| mean_point_delta | -6950.0000 [-8725.0000, -4912.5000] | 4462.5000 [-1956.2500, 10956.2500] | 11412.5000 [5750.0000, 18925.0000] |
| mean_rank | 2.6875 [2.3750, 2.9375] | 2.2500 [1.9375, 2.5000] | -0.4375 [-1.0000, -0.0625] |
| rank_1_rate | 18.75% [6.25%, 25.00%] | 31.25% [25.00%, 43.75%] | 12.50% [0.00%, 25.00%] |
| rank_2_rate | 18.75% [0.00%, 37.50%] | 25.00% [6.25%, 43.75%] | 6.25% [-31.25%, 25.00%] |
| rank_3_rate | 37.50% [25.00%, 50.00%] | 31.25% [6.25%, 62.50%] | -6.25% [-37.50%, 31.25%] |
| rank_4_rate | 25.00% [6.25%, 43.75%] | 12.50% [0.00%, 25.00%] | -12.50% [-37.50%, 12.50%] |

| Seed | Old rank | New rank | Rank difference | Old points | New points | Points difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 600500 | 2.75 | 2.50 | -0.25 | -8750 | -4425 | +4325 |
| 600501 | 2.25 | 2.25 | +0.00 | -3650 | 5450 | +9100 |
| 600502 | 2.75 | 2.50 | -0.25 | -6700 | 3325 | +10025 |
| 600503 | 3.00 | 1.75 | -1.25 | -8700 | 13500 | +22200 |

Parallel wall time: 1080.388 s.

baseline: process wall 1072.923 s; summed game time 1068.737 s; decisions 10957; candidate decisions 2793; illegal 0; candidate act mean/p95/max 0.301573/0.440083/11.492333 ms.
candidate: process wall 1080.383 s; summed game time 1075.854 s; decisions 10797; candidate decisions 2771; illegal 0; candidate act mean/p95/max 0.298527/0.445333/11.588167 ms.

- Four independent seed groups per model; 16 full hanchan each, not 32 independent samples
- Both historical models were trained with the previous native MJAI reader. A separate audit found blocked CHI inferred as PASS and missing second-player RON samples; this comparison does not measure performance after that reader correction
- Percentile bootstrap intervals with four groups are exploratory and highly discrete; 2000 resamples add no independent observations
- Akochan/efficiency/tsumogiri is a mixed opponent pool with fixed relative opponent order; this does not establish beating Akochan individually
- Candidate act latency excludes model loading and external transport; concurrent local resource contention prevents online timeout conclusions
- No multiple-comparison adjustment; mean rank is the primary registered metric, other metrics are descriptive
- Promotion sample size is not met; stable strength is not established
