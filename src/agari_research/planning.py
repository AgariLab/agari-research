"""Deterministic update budgets and per-game coverage derived from native counts."""

from __future__ import annotations

import hashlib
import math
import random

from agari_kit.mjai import canonical_json
from .trace_samples import SAMPLE_SEMANTICS_VERSION
from .selection import normalize_policy_filter

DEFAULT_STAGES = ("grp", "cql", "ppo")


def selected_stages(config: dict) -> tuple[str, ...]:
    stages = config.get("stages", list(DEFAULT_STAGES))
    if not isinstance(stages, list) or stages not in (["bc"], ["grp"], ["grp", "cql"], list(DEFAULT_STAGES)):
        raise ValueError("stages must be ['bc'], ['grp'], ['grp', 'cql'], or ['grp', 'cql', 'ppo']")
    return tuple(stages)


def sample_count_key(stage: str) -> str:
    # BC and CQL consume exactly the same selected, original behavior actions.
    return f"{'cql' if stage == 'bc' else stage}_samples"


def build_training_plan(config: dict, dataset) -> dict:
    active = selected_stages(config)
    selection = normalize_policy_filter(config.get("policy_filter"))
    if normalize_policy_filter(getattr(dataset, "policy_filter", None)) != selection:
        raise ValueError("Dataset policy_filter does not match training configuration")
    counts = dataset.sample_counts("train") if any("epochs" in config[stage] for stage in active if stage != "ppo") else []
    if counts:
        expected = [(record["game_id"], record["sha256"]) for record in dataset.records("train")]
        if [(row["game_id"], row["sha256"]) for row in counts] != expected:
            raise ValueError("Native sample counts do not match the frozen training games")
    stages = {}
    for stage in active:
        settings = config[stage]
        if stage != "ppo" and "epochs" in settings:
            column = sample_count_key(stage)
            total = sum(row[column] for row in counts)
            if not counts or any(type(row[column]) is not int or row[column] < 1 for row in counts):
                raise ValueError("Every training game must produce positive native sample counts")
            updates = math.ceil(total / settings["batch_size"])
            stages[stage] = {"mode": "epochs", "epochs": settings["epochs"], "samples_per_epoch": total,
                             "updates_per_epoch": updates, "target_updates": updates * settings["epochs"],
                             "target_samples": total * settings["epochs"],
                             "last_batch_samples": total % settings["batch_size"] or settings["batch_size"]}
        else:
            stages[stage] = {"mode": "steps", "target_updates": settings["steps"]}
            if stage != "ppo":
                stages[stage]["target_samples"] = settings["steps"] * settings["batch_size"]
    plan = {"schema_version": 1, "dataset_sha256": dataset.sha256,
            "sample_semantics_version": SAMPLE_SEMANTICS_VERSION,
            "producer_engines": getattr(dataset, "producer_engines", []),
            "policy_filter": selection,
            "training_games": counts, "stages": stages}
    plan["sha256"] = hashlib.sha256(canonical_json(plan).encode()).hexdigest()
    return plan


def expected_samples(stage_plan: dict, updates: int, batch_size: int) -> int:
    if not 0 <= updates <= stage_plan["target_updates"]:
        raise ValueError("Update cursor is outside the execution plan")
    if stage_plan["mode"] == "steps":
        return updates * batch_size
    epochs, within = divmod(updates, stage_plan["updates_per_epoch"])
    return epochs * stage_plan["samples_per_epoch"] + within * batch_size


def coverage(plan: dict, stage: str, seen: int, seed: int, *, include_games=False) -> dict:
    settings = plan["stages"][stage]
    result = {"mode": settings["mode"], "samples_seen": seen}
    if settings["mode"] != "epochs":
        return dict(result, complete_epochs=None, scope="fixed update budget; full coverage not asserted")
    if not 0 <= seen <= settings["target_samples"]:
        raise ValueError("Sample cursor is outside the execution plan")
    epochs, offset = divmod(seen, settings["samples_per_epoch"])
    rows = plan["training_games"]
    column = sample_count_key(stage)
    actual = {row["game_id"]: epochs * row[column] for row in rows}
    ordered = list(rows)
    random.Random(seed + epochs).shuffle(ordered)
    remaining = offset
    for row in ordered:
        used = min(remaining, row[column])
        actual[row["game_id"]] += used
        remaining -= used
        if not remaining:
            break
    result.update(complete_epochs=epochs, samples_in_epoch=offset, training_games=len(rows),
                  unique_games_seen=sum(value > 0 for value in actual.values()))
    if include_games:
        result["games"] = [{"game_id": row["game_id"], "sha256": row["sha256"],
                            "samples_per_epoch": row[column], "samples_seen": actual[row["game_id"]]}
                           for row in rows]
    return result
