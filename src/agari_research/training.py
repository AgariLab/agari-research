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
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from agari_kit.collection import write_json
from agari_kit.engine import require_engine
from agari_kit.mjai import canonical_json
from agari_kit.storage import Storage

from .data import BatchStream, EpochBatchStream, FrozenDataset
from .features import ENCODING_VERSION, UPSTREAM_COMMIT
from .losses import bc_loss, cql_loss, ppo_loss
from .manifest import file_sha256
from .models.actor_critic import ActorCriticNetwork
from .models.grp_model import RankPredictor
from .models.q_network import QNetwork
from .planning import build_training_plan, coverage, expected_samples, sample_count_key, selected_stages
from .rollout import collect_rollout
from .runtime import require_matching_runtime, runtime_provenance
from .trace_samples import SAMPLE_SEMANTICS_VERSION
from .selection import normalize_policy_filter

STAGES = ("grp", "cql", "ppo")


def implementation_sha256() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def validate_config(config: dict) -> None:
    stages = selected_stages(config)
    normalize_policy_filter(config.get("policy_filter"))
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
    if "grp" in stages and (len(config["pts_weight"]) != 4 or not all(math.isfinite(x) for x in config["pts_weight"])):
        raise ValueError("Four finite rank reward weights are required")
    for stage in stages:
        settings = config[stage]
        budgets = [key for key in ("steps", "epochs") if key in settings] if stage != "ppo" else ["steps"]
        if len(budgets) != 1:
            raise ValueError("BC/GRP/CQL require exactly one budget: steps or epochs")
        if any(type(settings.get(key)) is not int or settings[key] < 1 for key in (*budgets, "batch_size")):
            raise ValueError("Training budgets and batch sizes must be positive integers")
        if not 0 < settings["lr"] < 1:
            raise ValueError("Invalid learning rate")
        if stage != "ppo":
            eta_min = settings.get("eta_min", config.get("grp", {}).get("eta_min", 1e-7))
            if not math.isfinite(eta_min) or not 0 <= eta_min <= settings["lr"]:
                raise ValueError("Invalid minimum learning rate")
        if stage == "bc" and (not math.isfinite(settings.get("weight_decay", 0.1))
                              or settings.get("weight_decay", 0.1) < 0):
            raise ValueError("Invalid BC weight decay")
    if any(not 0 < config[stage]["gamma"] <= 1 for stage in stages if stage in ("cql", "ppo")):
        raise ValueError("Discount factors must be in (0, 1]")
    if "ppo" in stages and (config["ppo"]["hanchan_per_update"] < 1 or config["ppo"]["epochs"] < 1):
        raise ValueError("PPO requires positive complete-game and epoch counts")
    interval = config.get("checkpoint_interval_updates", 1)
    validation = config.get("validation", {})
    batch_size, limit = validation.get("batch_size", 128), validation.get("max_samples")
    if type(interval) is not int or interval < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Checkpoint interval and validation batch size must be positive integers")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Validation max_samples must be null or a positive integer")
    if "interval_updates" in validation and (type(validation["interval_updates"]) is not int
                                              or validation["interval_updates"] < 1):
        raise ValueError("Validation interval_updates must be a positive integer")


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


def initialize_q(config: dict, *, behavior_cloning: bool = False):
    """BC/CQL start from identical Q weights independently of other stages."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        model = QNetwork(**config["model"])
    if behavior_cloning:
        model.v_head.requires_grad_(False)
        if model.aux_head is not None:
            model.aux_head.requires_grad_(False)
    return model


def q_initialization_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(str(tuple(tensor.shape)).encode() + b"\0")
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def scalar_metrics(values: dict) -> dict:
    result = {key: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
              for key, value in values.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("Non-finite training metric; checkpoint not advanced")
    return result


def policy_diagnostics(model, dataset, config):
    """Report encoded-action accuracy by native action type and decision group."""
    factory = getattr(dataset, "policy_diagnostic_samples", None)
    if factory is None:
        return {"status": "unavailable", "reason": "dataset has no native decision diagnostics"}
    settings = config.get("validation", {})
    limit = settings.get("max_samples")
    samples = factory("validation")
    if limit is not None:
        samples = itertools.islice(samples, limit)
    samples = iter(samples)
    groups = {name: {"samples": 0, "correct": 0} for name in ("choice", "forced")}
    by_type = {}
    with torch.inference_mode():
        while rows := list(itertools.islice(samples, settings.get("batch_size", 128))):
            x, actions, masks = [torch.stack(values).to(config["device"]) for values in zip(*(row[:3] for row in rows))]
            scores = model(x)
            if not torch.isfinite(scores).all():
                raise RuntimeError("Non-finite policy validation scores")
            correct = scores.masked_fill(~masks, -torch.inf).argmax(1).eq(actions).cpu().tolist()
            for row, matched in zip(rows, correct):
                group = "forced" if row[4] else "choice"
                specific = by_type.setdefault(row[3], {name: {"samples": 0, "correct": 0}
                                                       for name in ("choice", "forced")})[group]
                for counter in (groups[group], specific):
                    counter["samples"] += 1
                    counter["correct"] += int(matched)
    for counter in [*groups.values(), *(counter for value in by_type.values() for counter in value.values())]:
        counter["accuracy"] = counter["correct"] / counter["samples"] if counter["samples"] else None
    return {"status": "complete", "metric": "encoded_action_accuracy", "scope": "full" if limit is None else "bounded",
            "sample_limit": limit, **groups, "by_action_type": by_type}


def validation_metrics(stage: str, dataset, models, config) -> dict:
    """Stream the held-out partition with sample-weighted metrics; never test."""
    model = models[stage]
    model.eval()
    if stage == "grp":
        factory = dataset.grp_samples("validation")
    elif stage == "bc":
        factory = dataset.bc_samples("validation")
    else:
        factory = dataset.cql_samples("validation", models["grp"], config)
    settings = config.get("validation", {})
    limit = settings.get("max_samples")
    if limit is not None:
        factory = itertools.islice(factory, limit)
    factory = iter(factory)
    totals, count = {}, 0
    with torch.inference_mode():
        while samples := list(itertools.islice(factory, settings.get("batch_size", 128))):
            batch = [torch.stack(values).to(config["device"]) for values in zip(*samples)]
            if stage == "grp":
                logits = model(batch[0])
                loss = F.cross_entropy(logits, batch[1])
                result = {"validation_loss": loss,
                          "validation_accuracy": (logits.argmax(1) == batch[1]).float().mean()}
            elif stage == "bc":
                loss, terms = bc_loss(model(batch[0]), *batch[1:])
                result = dict(validation_loss=loss, **{f"validation_{key}": value for key, value in terms.items()})
            else:
                q, aux = model.forward_with_aux(batch[0])
                loss, terms = cql_loss(q, aux, *batch[1:], alpha=config["cql"]["alpha"],
                                       aux_weight=config["cql"]["aux_weight"])
                result = dict(validation_loss=loss, **{f"validation_{key}": value for key, value in terms.items()})
                _, behavior = bc_loss(q, batch[1], batch[2])
                result.update({f"validation_{key}": value for key, value in behavior.items()})
            for key, value in scalar_metrics(result).items():
                totals[key] = totals.get(key, 0.0) + value * len(samples)
            count += len(samples)
    if not count:
        raise ValueError("Validation partition has no usable samples")
    result = dict({key: value / count for key, value in totals.items()}, validation_samples=count,
                  validation_scope="full" if limit is None else "bounded", validation_sample_limit=limit)
    if stage in ("bc", "cql"):
        diagnostics = policy_diagnostics(model, dataset, config)
        if diagnostics["status"] == "complete" and limit is None and diagnostics["choice"]["samples"] != count:
            raise RuntimeError("Validation diagnostics and training reader disagree on choice sample count")
        result["validation_policy_diagnostics"] = diagnostics
    return result


def run_training(config: dict, dataset: FrozenDataset, storage: Storage, *, run_name: str,
                 resume: Path | None = None, stop_after_updates: int | None = None,
                 max_wall_seconds: float | None = None) -> dict:
    started = time.monotonic()
    validate_config(config)
    stages = selected_stages(config)
    offline_stages = tuple(stage for stage in stages if stage != "ppo")
    require_engine()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_name):
        raise ValueError("Run name must be a simple path-safe identifier")
    if stop_after_updates is not None and stop_after_updates < 1:
        raise ValueError("stop_after_updates must be positive")
    if max_wall_seconds is not None and (type(max_wall_seconds) not in (int, float)
                                         or not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0):
        raise ValueError("max_wall_seconds must be finite and positive")
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
    runtime = runtime_provenance()
    restored = torch.load(resume, map_location="cpu", weights_only=True) if resume is not None else None
    if restored is not None:
        require_matching_runtime(restored.get("runtime_provenance"), runtime)
    plan = build_training_plan(config, dataset)
    models = {}
    if "grp" in stages:
        models["grp"] = RankPredictor()
    policy_kind = "bc" if "bc" in stages else "cql" if "cql" in stages else None
    if policy_kind:
        models[policy_kind] = initialize_q(config, behavior_cloning=policy_kind == "bc")
    policy_initialization = None if policy_kind is None else {
        "seed": config["seed"], "q_state_sha256": q_initialization_sha256(models[policy_kind])}
    if "ppo" in stages:
        models.update(ppo=ActorCriticNetwork(**config["model"]), reference=ActorCriticNetwork(**config["model"]))
    for model in models.values():
        model.to(config["device"])
    if "reference" in models:
        models["reference"].requires_grad_(False).eval()
    optimizers = {}
    for stage in stages:
        parameters = [parameter for parameter in models[stage].parameters() if parameter.requires_grad]
        settings = config[stage]
        optimizers[stage] = (torch.optim.Adam(parameters, lr=settings["lr"]) if stage == "grp" else
                             torch.optim.AdamW(parameters, lr=settings["lr"], weight_decay=settings.get("weight_decay", 0.1)))
    schedulers = {stage: torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[stage],
                  T_max=plan["stages"][stage]["target_updates"],
                  eta_min=config[stage].get("eta_min", config.get("grp", {}).get("eta_min", 1e-7)))
                  for stage in offline_stages}
    state = {"schema_version": 1, "config": config, "dataset_sha256": dataset.sha256,
             "sample_semantics_version": SAMPLE_SEMANTICS_VERSION,
             "implementation_sha256": implementation_sha256(), "encoding_version": ENCODING_VERSION,
             "engine_version": "0.4.10", "upstream_commit": UPSTREAM_COMMIT,
             "runtime_provenance": runtime, "training_plan": plan,
             "policy_filter": plan["policy_filter"], "policy_initialization": policy_initialization,
             "best_validation": {},
             "progress": {stage: 0 for stage in stages}, "samples_seen": {stage: 0 for stage in offline_stages},
             "ppo_initialized": False, "games_seen": 0, "run_name": run_name}
    if restored is not None:
        for key in ("schema_version", "config", "dataset_sha256", "implementation_sha256", "encoding_version",
                    "engine_version", "upstream_commit", "run_name", "training_plan", "sample_semantics_version",
                    "policy_filter", "policy_initialization"):
            if restored.get(key) != state[key]:
                raise ValueError(f"Resume checkpoint mismatch: {key}")
        for stage in offline_stages:
            expected = expected_samples(plan["stages"][stage], restored["progress"][stage], config[stage]["batch_size"])
            if restored["samples_seen"][stage] != expected:
                raise ValueError(f"Resume checkpoint mismatch: {stage} sample cursor")
        for name, model in models.items():
            model.load_state_dict(restored["models"][name], strict=True)
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(restored["optimizers"][name])
        for name, scheduler in schedulers.items():
            scheduler.load_state_dict(restored["schedulers"][name])
        restore_rng(restored["rng"])
        state = {key: value for key, value in restored.items() if key not in ("models", "optimizers", "schedulers", "rng")}
        if not isinstance(state.get("best_validation"), dict) or not set(state["best_validation"]) <= set(offline_stages):
            raise ValueError("Resume checkpoint mismatch: best_validation")
        for stage, selected in state["best_validation"].items():
            if (not isinstance(selected, dict) or selected.get("metric") != "validation_loss"
                    or type(selected.get("update")) is not int or not 1 <= selected["update"] <= state["progress"][stage]
                    or type(selected.get("value")) not in (int, float) or not math.isfinite(selected["value"])):
                raise ValueError("Resume checkpoint mismatch: best_validation selection")
            path = export_dir / selected["artifact"]
            if (path.parent != export_dir or not path.name.startswith(f"best-{stage}-u")
                    or file_sha256(path) != selected["sha256"]):
                raise ValueError("Resume checkpoint mismatch: selected validation model")
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "dataset.json", dataset.manifest)
    write_json(run_dir / "execution-plan.json", dict(plan=plan, runtime_provenance=runtime,
                                                   implementation_sha256=state["implementation_sha256"]))
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    checkpoint = checkpoint_dir / "latest.pt"
    invocation_updates = 0
    stop_reason = None

    def wall_budget_exhausted():
        return max_wall_seconds is not None and time.monotonic() - started >= max_wall_seconds

    def publish_best():
        # Versioned snapshots remain immutable; a checkpoint binds their hashes.
        # Rebuild the convenient alias from that checkpoint after an interruption.
        for stage, selected in state["best_validation"].items():
            source = export_dir / selected["artifact"]
            if file_sha256(source) != selected["sha256"]:
                raise ValueError("Selected validation model content changed")
            destination = export_dir / f"best-{stage}.pt"
            if not destination.exists() or file_sha256(destination) != selected["sha256"]:
                fd, temporary = tempfile.mkstemp(prefix=".best-", dir=export_dir)
                try:
                    with os.fdopen(fd, "wb") as output, source.open("rb") as input_stream:
                        shutil.copyfileobj(input_stream, output)
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temporary, destination)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            payload = torch.load(source, map_location="cpu", weights_only=True)
            metadata = {key: value for key, value in payload.items() if key != "state_dict"}
            metadata.update(weights_sha256=selected["sha256"], filename=destination.name)
            write_json(destination.with_suffix(".json"), metadata)
            if stage in ("bc", "cql"):
                write_agent_descriptor(stage, destination, selected["sha256"])

    def save():
        storage.check()
        payload = dict(state, models={name: model.state_dict() for name, model in models.items()},
                       optimizers={name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                       schedulers={name: scheduler.state_dict() for name, scheduler in schedulers.items()}, rng=rng_state())
        atomic_torch_save(payload, checkpoint)
        publish_best()
        write_json(run_dir / "coverage.json", {
            "dataset_sha256": dataset.sha256, "training_plan_sha256": plan["sha256"],
            "progress": state["progress"], "stages": {
                stage: coverage(plan, stage, state["samples_seen"][stage], config["seed"], include_games=True)
                for stage in offline_stages}})
        writer.flush()

    def write_agent_descriptor(stage, destination, digest):
        write_json(destination.with_name(destination.stem + "-agent.json"), {
            "factory": "agari_research.model_agent:ModelAgent",
            "kwargs": {"device": config["device"], "torch_threads": config["torch_threads"]},
            "artifacts": {"weights_path": {"path": destination.name, "sha256": digest}},
        })

    def export(stage, selection=None):
        name = f"{stage}.pt" if selection is None else f"best-{stage}-u{selection['update']:09d}.pt"
        destination = export_dir / name
        payload = {key: state[key] for key in ("schema_version", "dataset_sha256", "implementation_sha256",
                                             "encoding_version", "engine_version", "upstream_commit", "runtime_provenance",
                                             "sample_semantics_version", "policy_filter", "policy_initialization")}
        payload.update(kind=stage, model_config=config["model"], state_dict=models[stage].state_dict(),
                       config=config, completed_updates=state["progress"][stage], training_plan_sha256=plan["sha256"],
                       training_coverage={name: coverage(plan, name, state["samples_seen"][name], config["seed"])
                                          for name in offline_stages})
        if selection is not None:
            payload["validation_selection"] = selection
        # Checkpoint reloads can change Python object sharing without changing
        # values. Normalize export metadata and tensor storage so equivalent
        # uninterrupted/resumed models also have reproducible artifact hashes.
        weights = payload.pop("state_dict")
        payload = json.loads(canonical_json(payload))
        payload["state_dict"] = {key: tensor.detach().cpu().contiguous().clone()
                                 for key, tensor in weights.items()}
        if selection is not None and destination.exists():
            existing = torch.load(destination, map_location="cpu", weights_only=True)
            current_state = payload["state_dict"]
            old_state = existing.pop("state_dict")
            comparable = {key: value for key, value in payload.items() if key != "state_dict"}
            if (existing != comparable or set(old_state) != set(current_state)
                    or any(not torch.equal(old_state[key], current_state[key].detach().cpu()) for key in current_state)):
                raise ValueError("Existing validation snapshot differs; select a new run")
        else:
            atomic_torch_save(payload, destination)
        metadata = {key: value for key, value in payload.items() if key != "state_dict"}
        metadata.update(weights_sha256=file_sha256(destination), filename=destination.name)
        write_json(destination.with_suffix(".json"), metadata)
        if stage in ("bc", "cql", "ppo"):
            write_agent_descriptor(stage, destination, metadata["weights_sha256"])
        return {"artifact": destination.name, "sha256": metadata["weights_sha256"]}

    def log(stage, metrics):
        step = state["progress"][stage]
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical_json({"stage": stage, "update": step, **metrics}) + "\n")
        for key, value in metrics.items():
            if isinstance(value, (float, int)):
                writer.add_scalar(f"{stage}/{key}", value, step)

    try:
        if resume is not None:
            publish_best()
        for stage in stages:
            settings = config[stage]
            stage_plan = plan["stages"][stage]
            if state["progress"][stage] >= stage_plan["target_updates"]:
                continue
            if wall_budget_exhausted():
                stop_reason = "wall_time_budget"
                save()
                break
            if stage in offline_stages and stage_plan["mode"] == "epochs":
                factory = (dataset.grp_game_samples if stage == "grp" else dataset.bc_game_samples if stage == "bc"
                           else lambda record: dataset.cql_game_samples(record, models["grp"], config))
                stream = EpochBatchStream(dataset.records("train"),
                    {row["game_id"]: row[sample_count_key(stage)] for row in plan["training_games"]},
                    factory, settings["batch_size"], settings["epochs"], state["samples_seen"][stage], config["seed"])
            elif stage == "grp":
                stream = BatchStream(lambda seed: dataset.grp_samples("train", seed), settings["batch_size"],
                                     state["samples_seen"][stage], config["seed"])
            elif stage == "bc":
                stream = BatchStream(lambda seed: dataset.bc_samples("train", seed), settings["batch_size"],
                                     state["samples_seen"][stage], config["seed"])
            elif stage == "cql":
                models["grp"].eval()
                stream = BatchStream(lambda seed: dataset.cql_samples("train", models["grp"], config, seed),
                                     settings["batch_size"], state["samples_seen"][stage], config["seed"])
            elif not state["ppo_initialized"]:
                initialize_policy(models["cql"], models["ppo"], models["reference"])
                state["ppo_initialized"] = True
            while state["progress"][stage] < stage_plan["target_updates"]:
                if wall_budget_exhausted():
                    stop_reason = "wall_time_budget"
                    save()
                    break
                model, optimizer = models[stage], optimizers[stage]
                model.train()
                if stage in offline_stages:
                    batch = stream.next(config["device"])
                    optimizer.zero_grad(set_to_none=True)
                    if stage == "grp":
                        loss = F.cross_entropy(model(batch[0]), batch[1])
                        metrics = {}
                    elif stage == "bc":
                        loss, metrics = bc_loss(model(batch[0]), *batch[1:])
                    else:
                        q, aux = model.forward_with_aux(batch[0])
                        loss, metrics = cql_loss(q, aux, *batch[1:], alpha=settings["alpha"], aux_weight=settings["aux_weight"])
                    metrics = scalar_metrics(dict(metrics, loss=loss, lr=optimizer.param_groups[0]["lr"]))
                    loss.backward()
                    if stage in ("bc", "cql"):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
                    optimizer.step()
                    schedulers[stage].step()
                    state["samples_seen"][stage] = stream.seen
                    metrics.update(batch_samples=len(batch[0]),
                                   **coverage(plan, stage, stream.seen, config["seed"]))
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
                stage_complete = state["progress"][stage] == stage_plan["target_updates"]
                interval = config.get("validation", {}).get("interval_updates")
                validate_now = stage in offline_stages and (stage_complete or (
                    interval is not None and state["progress"][stage] % interval == 0))
                if validate_now:
                    metrics.update(validation_metrics(stage, dataset, models, config))
                    value = metrics["validation_loss"]
                    previous = state["best_validation"].get(stage)
                    if previous is None or value < previous["value"]:
                        selection = {"metric": "validation_loss", "value": value, "update": state["progress"][stage]}
                        state["best_validation"][stage] = dict(selection, **export(stage, selection))
                if stage_complete:
                    export(stage)
                log(stage, metrics)
                stopping = stop_after_updates is not None and invocation_updates >= stop_after_updates
                budget_exhausted = wall_budget_exhausted()
                epoch_complete = stage_plan["mode"] == "epochs" and stream.seen % stage_plan["samples_per_epoch"] == 0
                if (stage_complete or epoch_complete or validate_now or stopping or budget_exhausted or
                        sum(state["progress"].values()) % config.get("checkpoint_interval_updates", 1) == 0):
                    save()
                if budget_exhausted or stopping:
                    stop_reason = "wall_time_budget" if budget_exhausted else "update_limit"
                    break
            if stop_reason is not None:
                break
    finally:
        writer.close()
    complete = all(state["progress"][stage] == plan["stages"][stage]["target_updates"] for stage in stages)
    result = {"run": run_name, "status": "complete" if complete else "paused" if stop_reason == "wall_time_budget" else "interrupted",
              "stop_reason": None if complete else stop_reason,
              "invocation_elapsed_seconds": time.monotonic() - started,
              "best_validation": state["best_validation"],
              "progress": state["progress"], "checkpoint": str(checkpoint), "models": str(export_dir),
              "dataset_sha256": dataset.sha256, "implementation_sha256": state["implementation_sha256"],
              "training_plan_sha256": plan["sha256"],
              "training_coverage": {stage: coverage(plan, stage, state["samples_seen"][stage], config["seed"])
                                    for stage in offline_stages}}
    write_json(run_dir / "result.json", result)
    return result
