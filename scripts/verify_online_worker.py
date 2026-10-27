"""Local process/protocol preflight with real observations; never connects to a server."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
from collections import Counter

from agari_kit.agent import Context, Rules, ensure_legal, is_tsumogiri, make_agent
from agari_kit.collection import write_json
from agari_kit.engine import require_engine
from agari_kit.online import OnlineSession, clean_action
from agari_kit.plugins import agent_provenance
from agari_kit.provenance import source_fingerprint
from agari_kit.storage import Storage
from agari_kit.worker import ProcessRunner
from riichienv import GameRule, Observation, RiichiEnv


def legal_messages(observation, seat):
    messages = []
    for action in observation.legal_actions():
        message = clean_action(json.loads(action.to_mjai()))
        if message["type"] != "none":
            message["actor"] = seat
        if message["type"] == "dahai":
            message["tsumogiri"] = is_tsumogiri(action, observation)
        if message["type"] in ("chi", "pon", "daiminkan"):
            for event in reversed(observation.events):
                event = json.loads(event) if isinstance(event, str) else event
                if event["type"] == "dahai":
                    message["target"] = event["actor"]
                    break
        roundtrip = observation.select_action_from_mjai(message)
        if roundtrip is None or ensure_legal(roundtrip, observation).encode() != action.encode():
            raise AssertionError("Native legal action failed protocol round-trip")
        messages.append(message)
    return messages


def has_red(message):
    return any(tile in ("5mr", "5pr", "5sr") for tile in [message.get("pai"), *message.get("consumed", [])])


async def check_game(agent_name, seed, hero, budget_ms):
    rules = Rules(game_mode="4p-red-east")
    env = RiichiEnv(seed=seed, game_mode=rules.game_mode, rule=GameRule.default_tenhou())
    opponents = [make_agent("efficiency") for _ in range(4)]
    for seat, opponent in enumerate(opponents):
        opponent.reset(Context(seat, seed, rules))
    runner = ProcessRunner(agent_name)
    events, legal_types, chosen_types = [], Counter(), Counter()
    red_legal = red_chosen = requests = steps = 0
    session = OnlineSession(mode="validate", runner=runner, emit=events.append, seed=seed)
    try:
        await runner.start()
        await session.handle({"type": "start_game", "id": hero})
        observations = env.reset()
        while not env.done():
            if not observations or steps >= 100000:
                raise AssertionError("Local full-east game stalled")
            actions = {}
            for seat, observation in observations.items():
                possible = legal_messages(observation, seat)
                legal_types.update(message["type"] for message in possible)
                red_legal += sum(has_red(message) for message in possible)
                if seat == hero:
                    requests += 1
                    request = {"type": "request_action", "request_id": requests,
                               "observation": observation.serialize_to_base64(), "possible_actions": possible,
                               "time": {"grace_ms": budget_ms, "bank_ms": 0, "deadline_ms": budget_ms}}
                    response = await session.handle(request)
                    if response is None or response["request_id"] != requests:
                        raise AssertionError("Worker timed out or omitted a decision")
                    actions[seat] = ensure_legal(observation.select_action_from_mjai(response), observation)
                    chosen_types[response["type"]] += 1
                    red_chosen += has_red(response)
                    await session.handle({"type": "action_ack", "request_id": requests, "status": "accepted"})
                else:
                    actions[seat] = ensure_legal(opponents[seat].act(observation, Context(seat, seed, rules)), observation)
            observations = env.step(actions)
            steps += 1
        await session.handle({"type": "end_game", "scores": list(env.scores())})
        report = session.report()
        if report["counts"].get("local_timeouts", 0) or report["degraded"]:
            raise AssertionError("Worker was degraded or timed out")
        latencies = [event["latency_ms"] for event in events if event["event"] == "decision"]
        return {"seed": seed, "hero": hero, "local_game_completed": True, "steps": steps,
                "hands": sum(event["type"] == "end_kyoku" for event in env.mjai_log),
                "requests": requests, "median_latency_ms": statistics.median(latencies),
                "max_latency_ms": max(latencies), "all_players_legal_action_roundtrips": dict(legal_types),
                "all_players_red_action_roundtrips": red_legal,
                "worker_chosen_action_types": dict(chosen_types), "worker_red_actions": red_chosen,
                "session": report}
    finally:
        await runner.close()


async def check_forced_red_actions(agent_name, seed, budget_ms):
    """Find actual observations forcing a red discard, without editing state."""
    targets, rules = {}, Rules(game_mode="4p-red-east")
    for game in range(64):
        game_seed = seed + game
        env = RiichiEnv(seed=game_seed, game_mode=rules.game_mode, rule=GameRule.default_tenhou())
        agents = [make_agent("efficiency") for _ in range(4)]
        for seat, agent in enumerate(agents):
            agent.reset(Context(seat, game_seed, rules))
        observations, step = env.reset(), 0
        while not env.done():
            if not observations or step >= 100000:
                raise AssertionError("Red-action probe source game stalled")
            actions = {}
            for seat, observation in observations.items():
                legal = observation.legal_actions()
                if len(legal) == 1:
                    message = json.loads(legal[0].to_mjai())
                    tile = message.get("pai")
                    if message["type"] == "dahai" and tile in ("5mr", "5pr", "5sr") and tile not in targets:
                        targets[tile] = {"source_seed": game_seed, "source_step": step, "seat": seat,
                                         "observation": observation.serialize_to_base64(),
                                         "possible_actions": legal_messages(observation, seat)}
                actions[seat] = ensure_legal(agents[seat].act(observation, Context(seat, game_seed, rules)), observation)
            observations = env.step(actions)
            step += 1
        if len(targets) == 3:
            break
    if len(targets) != 3:
        raise AssertionError("Could not find all three real forced-red-discard observations within 64 east games")
    reports = []
    for tile, target in sorted(targets.items()):
        runner = ProcessRunner(agent_name)
        try:
            await runner.start()
            session = OnlineSession(mode="validate", runner=runner, emit=lambda event: None, seed=target["source_seed"])
            await session.handle({"type": "start_game", "id": target["seat"]})
            response = await session.handle({"type": "request_action", "request_id": 1,
                                             "observation": target["observation"], "possible_actions": target["possible_actions"],
                                             "time": {"grace_ms": budget_ms, "bank_ms": 0, "deadline_ms": budget_ms}})
            if response is None or response.get("pai") != tile or response["type"] != "dahai":
                raise AssertionError("Model worker failed a native forced red discard")
            observation = Observation.deserialize_from_base64(target["observation"])
            ensure_legal(observation.select_action_from_mjai(response), observation)
            reports.append({"tile": tile, "source_seed": target["source_seed"], "source_step": target["source_step"],
                            "seat": target["seat"], "legal_actions": 1, "worker_response": response,
                            "observation_sha256": hashlib.sha256(target["observation"].encode()).hexdigest()})
        finally:
            await runner.close()
    return reports


async def run(args):
    require_engine()
    storage = Storage.from_env()
    storage.check(probe=True)
    descriptor = storage.path(args.agent)
    agent_name = "@" + str(descriptor)
    provenance, kit = agent_provenance(agent_name), source_fingerprint()
    reports = []
    for index in range(args.games):
        reports.append(await check_game(agent_name, args.seed + index, index % 4, args.budget_ms))
        if provenance != agent_provenance(agent_name) or kit != source_fingerprint():
            raise AssertionError("Agent artifacts or code changed during the preflight")
    red_probes = await check_forced_red_actions(agent_name, args.seed + 10000, args.budget_ms) if args.targeted_red_probes else []
    if provenance != agent_provenance(agent_name) or kit != source_fingerprint():
        raise AssertionError("Agent artifacts or code changed during targeted probes")
    result = {"status": "local-preflight-complete", "network_connected": False,
              "server_validation": "not-performed", "acknowledgements": "locally simulated after native legality checks",
              "scope": "Real complete east games, persistent spawned worker and validate-mode request protocol",
              "agent": provenance, "kit": kit, "budget_ms": args.budget_ms,
              "forced_red_probes": red_probes,
              "games": reports, "completed_games": len(reports),
              "requests": sum(report["requests"] for report in reports), "illegal_actions": 0, "timeouts": 0}
    destination = storage.path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("Select a new preflight report path")
    write_json(destination, result)
    print(json.dumps({key: result[key] for key in ("status", "network_connected", "server_validation", "completed_games", "requests", "illegal_actions", "timeouts")}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, help="Storage-relative exported Agent descriptor")
    parser.add_argument("--output", required=True, help="New storage-relative JSON report")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--seed", type=int, default=88400)
    parser.add_argument("--budget-ms", type=int, default=1000)
    parser.add_argument("--targeted-red-probes", action="store_true")
    args = parser.parse_args()
    if args.games < 1 or args.budget_ms <= 100:
        parser.error("Use positive games and a budget above the 100ms reserve")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
