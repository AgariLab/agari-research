# CPU and MPS full-path training and resume checks

Completed 2026-10-29 using frozen kit `78719f35` and research `80d3c420`.
The [report](report.json) is an exact copy of the external execution record,
SHA-256 `4b4a9f0037e4779a2d574b8d7035275ecd398bcb5359907ee93c5e40a04a6de2`.
The separate [execution context](execution-context.json) records outer timing
and concurrent teacher qualification; it does not replace or amend the report.

| Device | Complete BC | Complete GRP/CQL | Exact resumed checkpoint tensors |
| --- | ---: | ---: | ---: |
| CPU | 10.04 s | 10.15 s | 149 / 185 |
| MPS | 9.39 s | 10.47 s | 149 / 185 |

Each method/device includes two interruptions, exact model/optimizer/scheduler/
RNG/cursor/best-selection comparison, identical final/best artifact hashes, and
a legal action through the public Agent. Equality is required within each device,
not between CPU and MPS. The frozen split contains 18 training games and one
validation game; the one held-out test game's content was not read.

Full command wall time was 89.36 seconds and cumulative peak process RSS was
658,456,576 bytes. Individual complete-training timings exclude separate data
preflight; interrupted runs, verification and exports are included in the full
command timing. These are concurrent small-task measurements, not idle-machine
throughput or strength results. They show no clear overall MPS speed advantage;
the first controlled A/B/C matrix therefore uses CPU uniformly.
