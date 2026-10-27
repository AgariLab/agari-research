"""Bounded single-process self-play with player-only policy observations."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from agari_kit.agent import ensure_legal
from agari_kit.engine import new_environment
from riichienv import MjaiReplay

from .features import action_mask, policy_features, predicted_rewards
from .losses import gae


def collect_rollout(policy, reference, grp, config: dict, *, seed: int, hero: int, temp_dir: Path):
    device, settings = config["device"], config["ppo"]
    policy.eval()
    reference.eval()
    grp.eval()
    env = new_environment(seed)
    observations = env.reset()
    hands, hand = [], []
    hand_count = steps = 0
    while not env.done():
        if not observations or steps >= 100000:
            raise RuntimeError("PPO self-play stalled or exceeded its full-game step limit")
        selected = {}
        for player, observation in observations.items():
            x, mask = policy_features(observation), action_mask(observation)
            model = policy if player == hero else reference
            with torch.inference_mode():
                logits, value = model(x.unsqueeze(0).to(device))
                masked = logits.squeeze(0).cpu().masked_fill(~mask, -1e9)
                if player == hero:
                    distribution = torch.distributions.Categorical(logits=masked)
                    action_id = distribution.sample()
                    hand.append((x, mask, action_id, distribution.log_prob(action_id), value.item()))
                else:
                    action_id = masked.argmax()
            selected[player] = ensure_legal(observation.find_action(int(action_id)), observation)
        observations = env.step(selected)
        steps += 1
        # kyoku_idx is unchanged on renchan. The explicit boundary is authoritative.
        ended = sum(event["type"] == "end_kyoku" for event in env.mjai_log)
        if ended != hand_count:
            if ended != hand_count + 1:
                raise RuntimeError("Unexpected native hand boundary")
            hands.append(hand)
            hand, hand_count = [], ended
    if hand:
        raise RuntimeError("PPO terminal hand did not close")
    # Use the same native reward feature extraction as offline CQL, including
    # riichi sticks, honba and multiple wins. This file is bounded to one game.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", dir=temp_dir) as stream:
        for event in env.mjai_log:
            stream.write(json.dumps(event) + "\n")
        stream.flush()
        replay = MjaiReplay.from_jsonl(stream.name, rule="tenhou")
        kyokus = list(replay.take_kyokus())
    if len(kyokus) != len(hands):
        raise RuntimeError("Native replay hand count differs from collected policy trajectories")
    samples = []
    for kyoku, trajectory in zip(kyokus, hands):
        reward = predicted_rewards(grp, kyoku.take_grp_features(), config["pts_weight"], device)[hero]
        advantages, returns = gae([step[4] for step in trajectory], reward,
                                  settings["gamma"], settings["gae_lambda"])
        for step, advantage, target in zip(trajectory, advantages, returns):
            samples.append((*step[:4], torch.tensor(advantage), torch.tensor(target)))
    if not samples:
        raise RuntimeError("PPO self-play produced no hero decisions")
    return samples, {"decisions": len(samples), "hands": len(hands), "steps": steps,
                     "scores": list(env.scores()), "hero": hero, "seed": seed}
