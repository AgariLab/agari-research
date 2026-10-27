"""Small-machine GRP → MC-CQL → PPO adapted from fixed RiichiEnv v0.4.10.

The model and loss lineage is Apache-2.0; see THIRD_PARTY_NOTICES.md. Training
artifacts and temporary replays live on explicitly configured project storage.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import re
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from agari_kit.collection import write_json
from agari_kit.engine import require_engine
from agari_kit.mjai import canonical_json
from agari_kit.storage import Storage

from .data import BatchStream, FrozenDataset
from .features import ENCODING_VERSION, UPSTREAM_COMMIT
from .losses import cql_loss, ppo_loss
from .manifest import file_sha256
from .models.actor_critic import ActorCriticNetwork
from .models.grp_model import RankPredictor
from .models.q_network import QNetwork
from .rollout import collect_rollout

STAGES = ("grp", "cql", "ppo")


def implementation_sha256() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def validate_config(config: dict) -> None:
    if config.get("schema_version") != 1 or config.get("device") not in ("cpu", "mps"):
        raise ValueError("Use configuration schema 1 and an explicit cpu or mps device")
    if config["device"] == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was selected but is unavailable; no implicit device fallback")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("Use an unsigned 32-bit training seed")
    if type(config["torch_threads"]) is not int or config["torch_threads"] < 1:
        raise ValueError("torch_threads must be positive")
    model = config["model"]
    if any(model.get(key) != expected for key, expected in
           (("in_channels", 74), ("num_actions", 82), ("tile_dim", 34), ("aux_dims", 4))):
        raise ValueError("The pinned upstream policy uses 74x34 features, 82 actions and four rank labels")
    if model["conv_channels"] < 16 or model["num_blocks"] < 1 or model["fc_dim"] < 1:
        raise ValueError("Invalid ResNet capacity")
    if len(config["pts_weight"]) != 4 or not all(math.isfinite(x) for x in config["pts_weight"]):
        raise ValueError("Four finite rank reward weights are required")
    for stage in STAGES:
        settings = config[stage]
        if any(type(settings[key]) is not int or settings[key] < 1 for key in ("steps", "batch_size")):
            raise ValueError("Training steps and batch sizes must be positive integers")
        if not 0 < settings["lr"] < 1:
            raise ValueError("Invalid learning rate")
    if any(not 0 < config[stage]["gamma"] <= 1 for stage in ("cql", "ppo")):
        raise ValueError("Discount factors must be in (0, 1]")
    if config["ppo"]["hanchan_per_update"] < 1 or config["ppo"]["epochs"] < 1:
        raise ValueError("PPO requires positive complete-game and epoch counts")


def rng_state() -> dict:
    state = np.random.get_state()
    result = {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
              "torch": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        result["mps"] = torch.mps.get_rng_state()
    return result


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def atomic_torch_save(value: dict, destination: Path) -> None:
    fd, name = tempfile.mkstemp(prefix=".checkpoint-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, destination)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def initialize_policy(q_network, policy, reference) -> None:
    state = q_network.state_dict()
    mapped = {}
    for key, value in state.items():
        if key.startswith("aux_head."):
            continue
        mapped[key.replace("a_head.", "actor_head.").replace("v_head.", "critic_head.")] = value
    policy.load_state_dict(mapped, strict=True)
    reference.load_state_dict(mapped, strict=True)
    reference.requires_grad_(False).eval()


def scalar_metrics(values: dict) -> dict:
    result = {key: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
              for key, value in values.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("Non-finite training metric; checkpoint not advanced")
    return result


def validation_metrics(stage: str, dataset, models, config) -> dict:
    """Fixed first 128 held-out samples, never the test partition."""
    model = models[stage]
    model.eval()
    factory = dataset.grp_samples("validation") if stage == "grp" else dataset.cql_samples(
        "validation", models["grp"], config)
    samples = list(itertools.islice(factory, 128))
    if not samples:
        raise ValueError("Validation partition has no usable samples")
    batch = [torch.stack(values).to(config["device"]) for values in zip(*samples)]
    with torch.inference_mode():
        if stage == "grp":
            logits = model(batch[0])
            loss = F.cross_entropy(logits, batch[1])
            result = {"validation_loss": loss, "validation_accuracy": (logits.argmax(1) == batch[1]).float().mean()}
        else:
            q, aux = model.forward_with_aux(batch[0])
            loss, _ = cql_loss(q, aux, *batch[1:], alpha=config["cql"]["alpha"],
                               aux_weight=config["cql"]["aux_weight"])
            result = {"validation_loss": loss}
    return dict(scalar_metrics(result), validation_samples=len(samples))


def run_training(config: dict, dataset: FrozenDataset, storage: Storage, *, run_name: str,
                 resume: Path | None = None, stop_after_updates: int | None = None) -> dict:
    validate_config(config)
    require_engine()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_name):
        raise ValueError("Run name must be a simple path-safe identifier")
    if stop_after_updates is not None and stop_after_updates < 1:
        raise ValueError("stop_after_updates must be positive")
    storage.initialize()
    os.environ.update(storage.environment())
    torch.set_num_threads(config["torch_threads"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["device"] == "cpu":
        torch.use_deterministic_algorithms(True)
    run_dir = storage.path(f"runs/{run_name}")
    checkpoint_dir, export_dir = storage.path(f"checkpoints/{run_name}"), storage.path(f"models/{run_name}")
    if resume is None and any(path.exists() for path in (run_dir, checkpoint_dir, export_dir)):
        raise FileExistsError("Run already exists; use an explicit checkpoint to resume or a new run name")
    for path in (run_dir, checkpoint_dir, export_dir):
        path.mkdir(parents=True, exist_ok=True)
    models = {"grp": RankPredictor(), "cql": QNetwork(**config["model"]),
              "ppo": ActorCriticNetwork(**config["model"]), "reference": ActorCriticNetwork(**config["model"])}
    for model in models.values():
        model.to(config["device"])
    models["reference"].requires_grad_(False).eval()
    optimizers = {
        "grp": torch.optim.Adam(models["grp"].parameters(), lr=config["grp"]["lr"]),
        "cql": torch.optim.AdamW(models["cql"].parameters(), lr=config["cql"]["lr"], weight_decay=config["cql"]["weight_decay"]),
        "ppo": torch.optim.AdamW(models["ppo"].parameters(), lr=config["ppo"]["lr"], weight_decay=config["ppo"]["weight_decay"]),
    }
    schedulers = {stage: torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[stage],
                  T_max=config[stage]["steps"], eta_min=config["grp"]["eta_min"])
                  for stage in ("grp", "cql")}
    state = {"schema_version": 1, "config": config, "dataset_sha256": dataset.sha256,
             "implementation_sha256": implementation_sha256(), "encoding_version": ENCODING_VERSION,
             "engine_version": "0.4.10", "upstream_commit": UPSTREAM_COMMIT,
             "progress": {stage: 0 for stage in STAGES}, "samples_seen": {stage: 0 for stage in ("grp", "cql")},
             "ppo_initialized": False, "games_seen": 0, "run_name": run_name}
    if resume is not None:
        restored = torch.load(resume, map_location="cpu", weights_only=True)
        for key in ("schema_version", "config", "dataset_sha256", "implementation_sha256", "encoding_version",
                    "engine_version", "upstream_commit", "run_name"):
            if restored.get(key) != state[key]:
                raise ValueError(f"Resume checkpoint mismatch: {key}")
        for name, model in models.items():
            model.load_state_dict(restored["models"][name], strict=True)
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(restored["optimizers"][name])
        for name, scheduler in schedulers.items():
            scheduler.load_state_dict(restored["schedulers"][name])
        restore_rng(restored["rng"])
        state = {key: value for key, value in restored.items() if key not in ("models", "optimizers", "schedulers", "rng")}
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "dataset.json", dataset.manifest)
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    checkpoint = checkpoint_dir / "latest.pt"
    invocation_updates = 0

    def save():
        storage.check()
        payload = dict(state, models={name: model.state_dict() for name, model in models.items()},
                       optimizers={name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                       schedulers={name: scheduler.state_dict() for name, scheduler in schedulers.items()}, rng=rng_state())
        atomic_torch_save(payload, checkpoint)

    def export(stage):
        destination = export_dir / f"{stage}.pt"
        payload = {key: state[key] for key in ("schema_version", "dataset_sha256", "implementation_sha256",
                                             "encoding_version", "engine_version", "upstream_commit")}
        payload.update(kind=stage, model_config=config["model"], state_dict=models[stage].state_dict(),
                       config=config, completed_updates=state["progress"][stage])
        atomic_torch_save(payload, destination)
        metadata = {key: value for key, value in payload.items() if key != "state_dict"}
        metadata.update(weights_sha256=file_sha256(destination), filename=destination.name)
        write_json(destination.with_suffix(".json"), metadata)
        if stage in ("cql", "ppo"):
            write_json(export_dir / f"{stage}-agent.json", {
                "factory": "agari_research.model_agent:ModelAgent",
                "kwargs": {"device": config["device"], "torch_threads": config["torch_threads"]},
                "artifacts": {"weights_path": {"path": destination.name, "sha256": metadata["weights_sha256"]}},
            })

    def log(stage, metrics):
        step = state["progress"][stage]
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical_json({"stage": stage, "update": step, **metrics}) + "\n")
        for key, value in metrics.items():
            if isinstance(value, (float, int)):
                writer.add_scalar(f"{stage}/{key}", value, step)
        writer.flush()

    try:
        for stage in STAGES:
            settings = config[stage]
            if state["progress"][stage] >= settings["steps"]:
                continue
            if stage == "grp":
                stream = BatchStream(lambda seed: dataset.grp_samples("train", seed), settings["batch_size"],
                                     state["samples_seen"][stage], config["seed"])
            elif stage == "cql":
                models["grp"].eval()
                stream = BatchStream(lambda seed: dataset.cql_samples("train", models["grp"], config, seed),
                                     settings["batch_size"], state["samples_seen"][stage], config["seed"])
            elif not state["ppo_initialized"]:
                initialize_policy(models["cql"], models["ppo"], models["reference"])
                state["ppo_initialized"] = True
            while state["progress"][stage] < settings["steps"]:
                model, optimizer = models[stage], optimizers[stage]
                model.train()
                if stage in ("grp", "cql"):
                    batch = stream.next(config["device"])
                    optimizer.zero_grad(set_to_none=True)
                    if stage == "grp":
                        loss = F.cross_entropy(model(batch[0]), batch[1])
                        metrics = {}
                    else:
                        q, aux = model.forward_with_aux(batch[0])
                        loss, metrics = cql_loss(q, aux, *batch[1:], alpha=settings["alpha"], aux_weight=settings["aux_weight"])
                    metrics = scalar_metrics(dict(metrics, loss=loss, lr=optimizer.param_groups[0]["lr"]))
                    loss.backward()
                    if stage == "cql":
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
                    optimizer.step()
                    schedulers[stage].step()
                    state["samples_seen"][stage] = stream.seen
                else:
                    samples, games = [], []
                    for _ in range(settings["hanchan_per_update"]):
                        game_index = state["games_seen"]
                        rows, report = collect_rollout(model, models["reference"], models["grp"], config,
                                                      seed=config["seed"] + 1000000 + game_index,
                                                      hero=game_index % 4, temp_dir=storage.path("tmp"))
                        samples.extend(rows)
                        games.append(report)
                        state["games_seen"] += 1
                    data = [torch.stack(values).to(config["device"]) for values in zip(*samples)]
                    data[4] = (data[4] - data[4].mean()) / (data[4].std() + 1e-8)
                    model.train()
                    for module in model.modules():
                        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                            module.eval()
                    metrics, count = {}, 0
                    warmup = settings["alpha_kl_warmup_steps"]
                    kl_coef = settings["alpha_kl"] * min(1.0, state["progress"][stage] / warmup) if warmup else settings["alpha_kl"]
                    for _ in range(settings["epochs"]):
                        permutation = torch.randperm(len(samples), device=config["device"])
                        for offset in range(0, len(samples), settings["batch_size"]):
                            indices = permutation[offset:offset + settings["batch_size"]]
                            x, masks, actions, old_log_probs, advantages, returns = [item[indices] for item in data]
                            optimizer.zero_grad(set_to_none=True)
                            logits, values = model(x)
                            with torch.no_grad():
                                reference_logits, _ = models["reference"](x)
                            loss, terms = ppo_loss(logits, values, reference_logits, masks, actions, old_log_probs,
                                                   advantages, returns, clip=settings["ppo_clip"],
                                                   value_coef=settings["value_coef"], entropy_coef=settings["entropy_coef"], kl_coef=kl_coef)
                            terms = scalar_metrics(dict(terms, loss=loss))
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"], error_if_nonfinite=True)
                            optimizer.step()
                            for key, value in terms.items():
                                metrics[key] = metrics.get(key, 0.0) + value
                            count += 1
                    metrics = {key: value / count for key, value in metrics.items()}
                    metrics.update(samples=len(samples), minibatches=count, kl_coef=kl_coef, games=games)
                state["progress"][stage] += 1
                invocation_updates += 1
                if state["progress"][stage] == settings["steps"]:
                    if stage in ("grp", "cql"):
                        metrics.update(validation_metrics(stage, dataset, models, config))
                    export(stage)
                log(stage, metrics)
                save()
                if stop_after_updates is not None and invocation_updates >= stop_after_updates:
                    break
            if stop_after_updates is not None and invocation_updates >= stop_after_updates:
                break
    finally:
        writer.close()
    complete = all(state["progress"][stage] == config[stage]["steps"] for stage in STAGES)
    result = {"run": run_name, "status": "complete" if complete else "interrupted",
              "progress": state["progress"], "checkpoint": str(checkpoint), "models": str(export_dir),
              "dataset_sha256": dataset.sha256, "implementation_sha256": state["implementation_sha256"]}
    write_json(run_dir / "result.json", result)
    return result
