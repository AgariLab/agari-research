# Phase 2: teacher baseline and student learning

Registered 2026-10-29 before model acquisition and all new games/training.
Status: implementation and CPU/MPS resume checks complete; teacher qualification
passed and the registered 1,000-game pilot collection is running. Student training
and independent promotion evaluation remain outstanding.

The immutable [protocol](protocol.json) records the teacher release, incumbent,
disjoint seed ranges, three training arms, selection rule and inference gates.
All runtime receipts and heavyweight artifacts use new `phase2-*` names under
the configured external storage root. Historical releases remain unchanged.

The first gate compares Mortal-S42 and the incumbent against three frozen
incumbents, using 250 complete four-seat seed groups per model. Successful
loading alone does not satisfy this gate. Four-teacher data collection begins
with a separate 16-game timing probe, then a frozen 1,000-game pilot.

The pilot compares efficiency-data BC, teacher-data BC and teacher-data CQL,
using the same small architecture and equal policy update budgets at three
training seeds. The complete validation partition selects checkpoints within
each method. Development match results choose the final candidate; independent
final evaluation uses 10,000 hanchan per model. Akochan remains an exploratory
opponent not used to produce training labels.

The 10,000-game teacher dataset is a separate conditional scale-up with fresh
seeds. It requires a teacher-trained pilot candidate to improve on the incumbent
in the development primary pool; inconclusive pilot results remain inconclusive.

The 48-hour limit applies to each invocation and pauses at a complete game,
seed group or optimizer update. Resume retains the experiment identity. CPU is
the baseline; MPS requires actual training and resume validation before use.
No paid compute or online matches are started by this protocol.

[CPU/MPS evidence](device-smoke/README.md) confirms exact same-device recovery
for BC and GRP/CQL. The first matrix uses CPU uniformly because the measured
small tasks show no clear overall MPS throughput advantage.

Frozen qualification runtime: kit `78719f35ea17d68e3a6fcfd04a2aa969a95142cc`,
research `80d3c4204d381a93a7f53d729fa59c6074ff1b42`. Both sources are isolated
from later documentation edits. `qualify_teacher.py` registers and verifies
the real protocol evidence before either 1,000-game side runs; its report stage
rechecks the sealed campaigns and paired confidence interval. Training config
preparation requires that completed qualification, exact four-teacher collection
identities and the pre-registered collection seeds.

[Independent teacher qualification](teacher-qualification/README.md) passed:
1,000 games per side, 250 paired seed groups, zero failed/unsealed attempts and
zero illegal actions. Mean rank is 1.579 for S42 versus 2.500 for the incumbent;
the paired difference is -0.921 with 95% CI [-0.978, -0.861]. This establishes
the registered fixed-pool teacher gate, not student promotion or online strength.

The execution tools are separately frozen at research
`b9ee0715c8e1c16ea811e3e37f2a8ebd89814671` in
`phase2-tools-20261029-v1/agari-research`. Their `--runtime-root` still selects
the original paired runtime above. These base execution tools passed 115 tests;
with the later standalone Akochan helper, the combined suite passes 136 tests.
No runtime source or qualification contract changed for these tools.

- [`collect_teacher_pilot.py`](../../scripts/collect_teacher_pilot.py) rechecks
  qualification before the registered 1,000-game collection, records each
  invocation, and publishes the verified split only after collection completes.
- [`run_prepared_training.py`](../../scripts/run_prepared_training.py) executes
  one of nine prepared jobs, with explicit recovery and complete process cost
  receipts. [Training commands and recovery](TRAINING_RUNBOOK.md) include
  separate GRP and CQL invocations for measuring their costs.
- [`phase2_evaluation_matrix.py`](../../scripts/phase2_evaluation_matrix.py)
  binds all nine completed models and both development pools before selection.
  A qualifying small matrix must complete the large matrix before final seeds
  are opened. One shared reservation binds the final candidate across matrices.
- [`phase2_akochan.py`](../../scripts/phase2_akochan.py) binds the final freeze,
  runs the disjoint 16+48 seed groups only after both initial campaigns pass,
  and recomputes pooled confidence intervals and latency percentiles from raw
  paired groups and samples. Its separate frozen checkout is
  `phase2-akochan-tools-20261029-v1/agari-research` at
  `db831239dacebf503d60e98a8ac1151e301bbc68`; pass the original tools research
  checkout above as `--tools-root`. No real Akochan evaluation has started.

The external `runs/phase2-pilot-1000-launch-v1/` driver records the handoff from
the completed qualification to `phase2-teacher-1000-v1`, then to the nine CPU
configs in `runs/phase2-pilot-1000-v1/`. Qualification verification exited 0 and
pilot collection began on 2026-10-29 at 02:27 Asia/Shanghai. It stops on a failed
gate or paused collection, and does not start training or development/final
games. Its source and step logs remain with that run.

Model source: [Mortal-S42 fixed publication](https://huggingface.co/haor/Mortal-S42/tree/524981f21a929e3c3b269f65af79b84163bd02ca).
Published file hashes must be checked against downloaded bytes; publisher
evaluation is not a substitute for local independent evaluation.
