# Phase 2: teacher baseline and student learning

Registered 2026-10-29 before model acquisition and all new games/training.
Status: implementation in progress; no teacher strength or student improvement claim yet.

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

Model source: [Mortal-S42 fixed publication](https://huggingface.co/haor/Mortal-S42/tree/524981f21a929e3c3b269f65af79b84163bd02ca).
Published file hashes must be checked against downloaded bytes; publisher
evaluation is not a substitute for local independent evaluation.
