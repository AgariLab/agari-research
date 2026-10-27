# Third-party source notices

`src/agari_research/models/{backbone,q_network,actor_critic,grp_model}.py` derive from
Kohei Ozaki and contributors' [RiichiEnv](https://github.com/smly/RiichiEnv),
commit `479c1faeb33d082965eef8198f63261a79c0fce3`, under Apache-2.0 (see `LICENSE`).
Original paths and SHA-256 values are recorded in `configs/upstream-files.json`.
Imports were changed to local relative imports, and attribution headers were added.
Architectures and parameter names are preserved, allowing shape-matched upstream checkpoints.

Local data and training code follows that commit's GRP, `MCDataset`, CQL loss,
`_ppo_worker.py` and `_ppo_learner.py` algorithm definitions. It uses a single CPU/MPS
process, fixed complete-game partitions, JSONL/TensorBoard logging, fail-closed data
validation and complete resumable checkpoints. Full hand boundaries use `end_kyoku`
events, including repeated dealer rounds, rather than relying only on a changing
round index. Default local network capacity is smaller than the reference configuration.
These local adaptations and their tests are original AgariLab Apache-2.0 code.

No Mortal implementation or model weights are included.
