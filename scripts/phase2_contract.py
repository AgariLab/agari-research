"""Read-only gates shared by phase-two qualification and training preparation."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path

from agari_kit.agent import Rules, managed_agents
from agari_kit.collection import file_sha256
from agari_kit.evaluation_campaign import CONDITIONS, _identity, verify_campaign
from agari_kit.mjai import canonical_json
from agari_kit.plugins import agent_provenance
from agari_kit.validation import _trace_records

ROOT = Path(__file__).resolve().parents[1]
TEACHER_FACTORY = "agari_kit.opponents.mortal:MortalAgent"
NATIVE_CASES = {
    "test_ankan_dora_rinshan", "test_kakan_delayed_dora_chankan", "test_kyushu_decision_and_new_hand",
    "test_reach_followup_discard", "test_red_five_chi_followup_discard",
    "test_red_five_pon_followup_discard", "test_two_ron_events_then_next_hand",
}
EVALUATION_SEEDS = ("teacher_qualification", "teacher_probe", "development", "akochan_exploratory", "final_promotion")


def content_sha256(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def storage_relative(storage, path):
    path = Path(path).resolve()
    try:
        relative = path.relative_to(storage.root.resolve()).as_posix()
    except ValueError:
        raise ValueError("Phase-two evidence must remain on configured artifact storage") from None
    storage.path(relative)
    return relative


def seed_range(config):
    start, count = config["start"], config["count"]
    if type(start) is not int or type(count) is not int or start < 0 or count < 1 or start + count > 2**64:
        raise ValueError("Invalid registered seed range")
    return set(range(start, start + count))


def load_protocol(path):
    protocol = json.loads(Path(path).read_bytes())
    training = protocol["training"]
    expected_arms = [{"id": "A", "data": "existing-efficiency", "stages": ["bc"]},
                     {"id": "B", "data": "four-seat-teacher", "stages": ["bc"]},
                     {"id": "C", "data": "four-seat-teacher", "stages": ["grp", "cql"]}]
    if (protocol.get("schema_version") != 1 or protocol.get("experiment") != "phase2-teacher-student-v1"
            or protocol.get("rule_preset") != "4p-red-half/tenhou" or protocol.get("engine_version") != "0.4.10"
            or protocol.get("max_wall_seconds_per_invocation") != 172800
            or training["arms"] != expected_arms or training["seeds"] != [2026, 2027, 2028]
            or training["policy_batch_size"] != 64
            or training["policy_updates"] != "ceil(3 * max(weak_training_decisions, teacher_training_decisions) / 64)"
            or training["split"] != [0.9, 0.05, 0.05] or training["ppo_in_scope"] is not False
            or training["grp_cost_separate"] is not True):
        raise ValueError("Unsupported phase-two protocol; update implementation before changing its contract")
    seen = set()
    for name, values in protocol["seeds"].items():
        seeds = seed_range(values)
        if seen & seeds:
            raise ValueError("Registered seed ranges overlap")
        seen.update(seeds)
        if values["rotations"] != (1 if name in ("teacher_probe", "teacher_1000", "teacher_10000") else 4):
            raise ValueError("Unexpected registered seat rotation count")
    if (protocol["seeds"]["teacher_qualification"]["count"] != 250
            or protocol["seeds"]["teacher_1000"]["count"] != 1000
            or protocol["seeds"]["teacher_10000"]["count"] != 10000):
        raise ValueError("Unexpected teacher qualification or dataset size")
    evaluation = protocol["evaluation"]
    if (evaluation["qualification_opponents"] != ["incumbent"] * 3
            or evaluation["primary_metric"] != "candidate_minus_incumbent_mean_rank"
            or evaluation["inference"] != "paired-complete-seed-group-bootstrap-95pct"
            or evaluation["teacher_condition"] != "1000 games per model and CI upper bound < 0; adapter legality checks pass"):
        raise ValueError("Unsupported teacher qualification contract")
    return protocol


def comparison_module():
    path = ROOT / "scripts/compare_evaluations.py"
    spec = importlib.util.spec_from_file_location("phase2_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_teacher(storage, descriptor, protocol):
    descriptor = Path(descriptor).resolve()
    storage_relative(storage, descriptor)
    name = "@" + str(descriptor)
    provenance = agent_provenance(name)
    if provenance.get("factory") != TEACHER_FACTORY or set(provenance.get("artifacts", {})) != {"manifest_path"}:
        raise ValueError("Registered teacher must use the exact Mortal adapter and runtime manifest")
    manifest_path = Path(provenance["artifacts"]["manifest_path"]["path"])
    storage_relative(storage, manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    published = protocol["teacher"]
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "mortal-runtime-v1"
            or manifest.get("checkout") != "source"
            or manifest.get("upstream", {}).get("repo") != published["repository"]
            or manifest["upstream"].get("revision") != published["revision"]
            or manifest["upstream"].get("mortal_commit") != "0cff2b52982be5b1163aa9a62fb01f03ce91e0d2"):
        raise ValueError("Teacher runtime differs from the registered publication")
    source = manifest_path.parent / "source"
    identity_path = source / "model-identity.json"
    identity = json.loads(identity_path.read_bytes())
    if (identity.get("name") != "Mortal-S42" or identity.get("format") != "pytorch-state-dicts"
            or identity.get("file") != published["weights"] or identity.get("sha256") != published["sha256"]
            or identity.get("bytes") != published["bytes"] or identity.get("is_oracle") is not False
            or identity.get("observation_version") != 4 or identity.get("observation_shape") != [1012, 34]
            or identity.get("action_space") != 46
            or identity.get("resnet") != {"conv_channels": 192, "num_blocks": 40}):
        raise ValueError("Actual teacher model-identity differs from registered S42")
    weight = source / identity["file"]
    storage_relative(storage, weight)
    files = manifest.get("files", {})
    if (files.get("source/" + published["weights"]) != published["sha256"]
            or weight.stat().st_size != published["bytes"] or file_sha256(weight) != published["sha256"]
            or files.get("source/model-identity.json") != file_sha256(identity_path)):
        raise ValueError("Teacher weight or model-identity content hash mismatch")
    with managed_agents([name]):
        pass  # Constructor verifies all runtime files; no reset/worker/game.
    if agent_provenance(name) != provenance:
        raise ValueError("Teacher changed while verifying the external bundle")
    return provenance, {"descriptor": storage_relative(storage, descriptor),
                        "manifest_sha256": file_sha256(manifest_path),
                        "model_identity_sha256": file_sha256(identity_path),
                        "weights_sha256": published["sha256"]}


def verify_legality(storage, path, teacher, engine, rules):
    path = storage.path(path) if isinstance(path, str) else Path(path)
    relative = storage_relative(storage, path)
    report = json.loads(path.read_bytes())
    if (report.get("schema_version") != 1 or report.get("status") != "passed"
            or report.get("teacher_provenance") != teacher
            or report.get("teacher_provenance_sha256") != content_sha256(teacher)
            or report.get("engine") != engine or report.get("rules") != rules):
        raise ValueError("Teacher legality report identity or status mismatch")
    checks = report.get("checks")
    if not isinstance(checks, list) or not all(isinstance(check, dict) for check in checks):
        raise ValueError("Teacher legality checks are missing")
    names = [check.get("name") for check in checks]
    required = NATIVE_CASES | {"four-seat-native-half-games"} | {
        f"seat-{seat}-{suffix}" for seat in range(4) for suffix in ("mjai.jsonl.gz", "trace.jsonl.gz")}
    if len(set(names)) != len(names) or set(names) != required:
        raise ValueError("Teacher legality report lacks the required complete scenario set")
    indexed, payloads = {}, {}
    for check in checks:
        evidence = storage.path(check["evidence_path"])
        if check.get("status") != "passed" or file_sha256(evidence) != check.get("evidence_sha256"):
            raise ValueError("Teacher legality evidence hash or status mismatch")
        indexed[check["name"]] = check
        if check["name"] in NATIVE_CASES | {"four-seat-native-half-games"}:
            payloads[check["name"]] = json.loads(evidence.read_bytes())
    for name, payload in payloads.items():
        if (payload.get("status") != "passed" or payload.get("teacher_provenance") != teacher
                or payload.get("teacher_provenance_after") != teacher
                or payload.get("engine") != engine or payload.get("rules") != rules):
            raise ValueError("Underlying teacher legality probe disagrees with summary")
        if name in NATIVE_CASES and payload.get("case_status", {}).get(name) is not True:
            raise ValueError("Underlying teacher native scenario did not pass")
    games = payloads["four-seat-native-half-games"].get("games", [])
    if len(games) != 4 or {row.get("seat") for row in games} != set(range(4)):
        raise ValueError("Teacher legality probe must cover all four seats")
    for row in games:
        receipt = row.get("validation", {})
        if row.get("illegal_actions") != 0 or receipt.get("validation") != "engine-verified":
            raise ValueError("Teacher half-game lacks native legality verification")
        for suffix, field in (("mjai.jsonl.gz", "mjai_file_sha256"), ("trace.jsonl.gz", "trace_sha256")):
            if indexed[f"seat-{row['seat']}-{suffix}"]["evidence_sha256"] != receipt.get(field):
                raise ValueError("Teacher game evidence is not bound to its native receipt")
    return {"path": relative, "sha256": file_sha256(path), "checks": sorted(names)}


def registration_for(storage, protocol_path, teacher_descriptor, legality_path):
    protocol = load_protocol(protocol_path)
    teacher_provenance, binding = validate_teacher(storage, teacher_descriptor, protocol)
    base = "@" + str(storage.path(protocol["incumbent"]["descriptor"]))
    teacher = "@" + str(Path(teacher_descriptor).resolve())
    lineups = {"baseline": [base] * 4, "teacher": [teacher, base, base, base]}
    seeds = sorted(seed_range(protocol["seeds"]["teacher_qualification"]))
    identities = {name: dict(_identity(lineup), seeds=seeds) for name, lineup in lineups.items()}
    comparator = comparison_module()
    if comparator.comparison_conditions(identities["baseline"]) != comparator.comparison_conditions(identities["teacher"]):
        raise ValueError("Teacher qualification must share conditions and opponents")
    if (identities["teacher"]["lineup_provenance"][0] != teacher_provenance
            or identities["baseline"]["lineup_provenance"][0]["artifacts"]["weights_path"]["sha256"]
            != protocol["incumbent"]["weights_sha256"]):
        raise ValueError("Teacher or incumbent differs from registered identity")
    legality = verify_legality(storage, legality_path, teacher_provenance,
                              identities["teacher"]["engine"], identities["teacher"]["rules"])
    contract = {"schema_version": 1, "kind": "different-candidates",
                "hypothesis": "Frozen Mortal-S42 improves mean rank over the incumbent against three incumbent opponents",
                "primary_metric": "mean_rank",
                "baseline_provenance_sha256": comparator.identity_sha256(identities["baseline"]["lineup_provenance"][0]),
                "candidate_provenance_sha256": comparator.identity_sha256(teacher_provenance),
                "conditions_sha256": comparator.identity_sha256(comparator.comparison_conditions(identities["baseline"]))}
    registration = {"schema_version": 1, "protocol_sha256": file_sha256(protocol_path),
                    "runner_sha256": file_sha256(ROOT / "experiments/phase2-teacher-student-v1/qualify_teacher.py"),
                    "contract_helper_sha256": file_sha256(Path(__file__)),
                    "comparator_sha256": file_sha256(Path(comparator.__file__)), "teacher_binding": binding,
                    "legality": legality, "identities": identities, "contract": contract}
    return registration, protocol, comparator


def verify_registration(output, expected):
    if (json.loads((output / "registration.json").read_bytes()) != expected
            or json.loads((output / "comparison-contract.json").read_bytes()) != expected["contract"]):
        raise ValueError("Registered qualification identities, evidence or implementation changed")


def qualification_result(output, registration, comparator):
    reports = {name: verify_campaign(output / name) for name in ("baseline", "teacher")}
    for name in reports:
        manifest = json.loads((output / name / "manifest.json").read_bytes())
        if any(manifest.get(key) != registration["identities"][name][key] for key in (*CONDITIONS, "seeds")):
            raise ValueError("Qualification results differ from registered fixtures")
    comparison = comparator.compare(output / "baseline", output / "teacher", contract=registration["contract"])
    acceptable = all(report["attempts"]["failed"] == 0 and report["attempts"]["unsealed"] == 0
                     for report in reports.values())
    passed = comparison["games_per_model"] == 1000 and comparison["ci95"]["mean_rank"][1] < 0 and acceptable
    return {"schema_version": 1, "status": "passed" if passed else "not-qualified",
            "scope": "Fixed-pool teacher strength and bound native adapter legality checks",
            "no_failed_or_unsealed_attempts": acceptable, "comparison": comparison,
            "legality": registration["legality"],
            "registration_sha256": file_sha256(output / "registration.json")}


def require_qualification(storage, output, protocol_path, teacher_descriptor):
    registered = json.loads((output / "registration.json").read_bytes())
    registration, protocol, comparator = registration_for(
        storage, protocol_path, teacher_descriptor, registered["legality"]["path"])
    verify_registration(output, registration)
    result = qualification_result(output, registration, comparator)
    if (result["status"] != "passed" or json.loads((output / "qualification.json").read_bytes()) != result):
        raise ValueError("Teacher did not pass the independently recomputed qualification")
    return {"run": storage_relative(storage, output), "qualification_sha256": file_sha256(output / "qualification.json"),
            "registration_sha256": file_sha256(output / "registration.json"), "legality": registration["legality"],
            "collection_identity": {key: registration["identities"]["teacher"][key] for key in ("code", "engine", "rules")}}


def audit_source(storage, manifest, *, name, protocol, teacher_provenance=None, collection_identity=None):
    """Validate every receipt, but only open train/validation trace headers."""
    records = [(part, row) for part in ("train", "validation", "test") for row in manifest["partitions"][part]]
    count = len(records)
    if count not in (1000, 10000):
        raise ValueError("Phase-two sources must contain exactly 1000 or 10000 complete games")
    expected_players = ([teacher_provenance] * 4 if name == "teacher" else
                        [{"kind": "builtin", "name": "efficiency"}] * 4)
    expected_hashes = [content_sha256(value) for value in expected_players]
    parents = {storage.path(row["validation_receipt"]).parent for _, row in records}
    if len(parents) != 1:
        raise ValueError("A source must belong to one frozen acquisition")
    parent = parents.pop()
    acquisition_path = parent / "acquisition.json"
    acquisition = json.loads(acquisition_path.read_bytes())
    config = acquisition["config"]
    if name == "teacher":
        if (not isinstance(teacher_provenance, dict) or not isinstance(collection_identity, dict)
                or any(config.get(key) != collection_identity.get(key) for key in ("code", "engine", "rules"))):
            raise ValueError("Teacher acquisition must use the qualified producer, engine and rules")
        allowed = seed_range(protocol["seeds"][f"teacher_{count}"])
    else:
        old_name, start = ("selfplay-v1", 0) if count == 1000 else ("selfplay-10000-v1", 500000)
        if parent != storage.path("datasets/manifests/" + old_name):
            raise ValueError("Weak data must use the existing frozen efficiency acquisition")
        allowed = set(range(start, start + count))
    if (acquisition.get("status") != "complete" or config.get("source") != "agarilab-selfplay"
            or config.get("games") != count or config.get("seed_start") != min(allowed)
            or config.get("rules") != asdict(Rules()) or config.get("agent_provenance") != expected_players
            or config.get("agents") != [player["name"] for player in expected_players]):
        raise ValueError("Source acquisition identity, cohort or completion mismatch")
    seen, inventory = set(), []
    legacy_test = 0
    for partition, row in records:
        receipt_path = storage.path(row["validation_receipt"])
        if file_sha256(receipt_path) != row["validation_receipt_sha256"]:
            raise ValueError("Source receipt hash mismatch")
        receipt = json.loads(receipt_path.read_bytes())
        seed = receipt.get("seed")
        if (type(seed) is not int or seed not in allowed or seed in seen
                or receipt.get("validation") != "engine-verified" or receipt.get("method") != "seeded-action-trace-v1"
                or receipt.get("rules") != asdict(Rules()) or receipt.get("sha256") != row["sha256"]
                or receipt.get("engine_version") != protocol["engine_version"]
                or receipt.get("producer_code") != config["code"]):
            raise ValueError("Source receipt seed, rules or producer mismatch")
        if name == "teacher" and receipt.get("engine_binary_sha256") != collection_identity["engine"]["binary_sha256"]:
            raise ValueError("Teacher receipt engine differs from qualification")
        seen.add(seed)
        hashes = receipt.get("agent_provenance_sha256")
        if name == "teacher" or hashes is not None:
            if hashes != expected_hashes:
                raise ValueError("Source receipt does not bind all four expected players")
        elif partition == "test":
            legacy_test += 1
        if partition != "test":
            stream = iter(_trace_records(storage.path(row["trace_path"])))
            try:
                _, header = next(stream)
            finally:
                stream.close()
            if (header.get("seed") != seed or header.get("agent_provenance") != expected_players
                    or header.get("agents") != config["agents"] or header.get("code") != config["code"]
                    or header.get("rules") != config["rules"]):
                raise ValueError("Actual training trace header disagrees with source identity")
        inventory.append({"game_id": row["game_id"], "partition": partition, "seed": seed,
                          "receipt": row["validation_receipt"], "sha256": row["validation_receipt_sha256"]})
    if seen != allowed:
        raise ValueError("Source is missing registered collection seeds")
    reserved = set().union(*(seed_range(protocol["seeds"][key]) for key in EVALUATION_SEEDS))
    if seen & reserved:
        raise ValueError("Evaluation or probe seeds cannot enter training sources")
    return {"name": name, "games": count, "seeds": sorted(seen),
            "acquisition": storage_relative(storage, acquisition_path), "acquisition_sha256": file_sha256(acquisition_path),
            "receipt_inventory": inventory, "receipt_inventory_sha256": content_sha256(inventory),
            "legacy_test_games_without_per_game_policy_binding": legacy_test,
            "test_partition_content_opened": False,
            "test_policy_identity_evidence": "acquisition config only for legacy receipts; per-seat receipt hashes otherwise"}
