"""Player decisions from verified seeded traces, including overridden claims.

MJAI records executed calls, so it cannot distinguish a voluntary pass from a
claim overridden by another player. Never infer CQL decisions from that log.
Only one game's native state and one round's player observations are retained.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from agari_kit.agent import Rules
from agari_kit.engine import new_environment
from agari_kit.mjai import DataError, canonical_json
from agari_kit.validation import _trace_records
from riichienv import MjaiReplay

SAMPLE_SEMANTICS_VERSION = "seeded-trace-player-decisions-v1"


@dataclass(frozen=True)
class Decision:
    step: int
    observation: object
    action: object


@dataclass(frozen=True)
class TraceRound:
    features: dict
    players: list[list[Decision]]


def round_key(observation) -> tuple[int, int, int]:
    return observation.round_wind, observation.kyoku_index, observation.honba


def terminal_round_scores(engine) -> list[int]:
    """One terminal log read separates round settlement from match pot payout.

The engine awards unclaimed riichi deposits to the match leader at end_game;
the upstream GRP label intentionally describes the preceding round settlement.
"""
    events = engine.mjai_log
    start = next(index for index in range(len(events) - 1, -1, -1)
                 if events[index]["type"] == "start_kyoku")
    scores = list(events[start]["scores"])
    for event in events[start + 1:]:
        if event["type"] == "reach_accepted":
            scores[event["actor"]] -= 1000
        elif event["type"] in ("hora", "ryukyoku"):
            deltas = event.get("deltas", [0, 0, 0, 0])
            scores = [score + delta for score, delta in zip(scores, deltas)]
    return scores


def trace_rounds(mjai_path: Path, trace_path: Path, *, producer: dict,
                 max_steps: int = 100000, include_forced: bool = False):
    """Yield actual non-forced decisions grouped by native, checked round results.

The caller must verify archive/receipt hashes and complete engine replay first.
The pinned kit parser bounds each JSON record; the trace is streamed, while the
    native MJAI parser retains at most this one game's result features. The full
    native log is read only once, at game end, to separate final-pot settlement.
"""
    records = iter(_trace_records(trace_path))
    try:
        _, header = next(records)
    except StopIteration as exc:
        raise DataError("CQL requires a nonempty seeded action trace") from exc
    if (header.get("type") != "agari_trace" or type(header.get("schema_version")) is not int
            or header["schema_version"] != 1
            or header.get("rules") != asdict(Rules()) or header.get("code") != producer
            or type(header.get("seed")) is not int or not 0 <= header["seed"] < 2**64
            or not isinstance(header.get("agents"), list) or len(header["agents"]) != 4):
        raise DataError("Unsupported seeded trace header, producer or rules")
    kyokus = iter(MjaiReplay.from_jsonl(str(mjai_path), rule="tenhou").take_kyokus())
    engine = new_environment(header["seed"])
    observations = engine.reset()
    features = key = None
    players = [[], [], [], []]
    steps = 0
    footer = None
    for line, record in records:
        where = f"{trace_path}:{line}"
        if record.get("type") == "trace_end":
            footer = record
            if next(records, None) is not None:
                raise DataError(f"{where}: records after trace_end")
            break
        if engine.done() or not observations or steps >= max_steps:
            raise DataError(f"{where}: unexpected decision or replay step limit")
        if type(record.get("step")) is not int or record["step"] != steps:
            raise DataError(f"{where}: non-contiguous trace step")
        first = next(iter(observations.values()))
        current_key = round_key(first)
        if first.honba == 255:
            raise DataError("Saturated native honba cannot safely identify consecutive rounds")
        if any(round_key(obs) != current_key for obs in observations.values()):
            raise DataError(f"{where}: active players disagree on the round")
        if features is None:
            kyoku = next(kyokus, None)
            if kyoku is None:
                raise DataError(f"{where}: trace has more rounds than MJAI")
            features = kyoku.take_grp_features()
            key = (features["chang"], features["ju"], features["ben"])
            if (current_key != key or list(first.scores) != features["round_initial_scores"]
                    or first.riichi_sticks != features["liqibang"]):
                raise DataError(f"{where}: native trace and GRP round start disagree")
        elif current_key != key:
            raise DataError(f"{where}: unclosed trace round")
        submitted, hashes = record.get("actions"), record.get("observations_sha256")
        seats = {str(seat) for seat in observations}
        if (not isinstance(submitted, dict) or set(submitted) != seats
                or not isinstance(hashes, dict) or set(hashes) != seats):
            raise DataError(f"{where}: trace seats disagree with native observations")
        actions = {}
        for seat, observation in observations.items():
            encoded = observation.serialize_to_base64()
            if hashlib.sha256(encoded.encode()).hexdigest() != hashes[str(seat)]:
                raise DataError(f"{where}: native Observation hash mismatch")
            legal = observation.legal_actions()
            matches = [action for action in legal
                       if canonical_json(action.to_dict()) == canonical_json(submitted[str(seat)])]
            if len(matches) != 1:
                raise DataError(f"{where}: submitted action is not uniquely native-legal")
            actions[seat] = matches[0]
            if len(legal) > 1 or include_forced:
                players[seat].append(Decision(steps, observation, matches[0]))
        observations = engine.step(actions)
        steps += 1
        if engine.done() or (observations and round_key(next(iter(observations.values()))) != key):
            scores = terminal_round_scores(engine) if engine.done() else list(engine.scores())
            if (scores != features["round_end_scores"] or
                    [end - start for start, end in zip(features["round_initial_scores"], scores)]
                    != features["round_delta_scores"]):
                raise DataError(f"{where}: native trace and GRP round result disagree")
            yield TraceRound(features, players)
            features, key, players = None, None, [[], [], [], []]
    if (footer is None or not engine.done() or features is not None
            or type(footer.get("steps")) is not int or footer["steps"] != steps
            or footer.get("scores") != list(engine.scores())):
        raise DataError("Incomplete seeded trace or incorrect terminal result")
    if next(kyokus, None) is not None:
        raise DataError("MJAI has more rounds than the seeded trace")
