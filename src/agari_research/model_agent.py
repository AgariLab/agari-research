"""Exported neural policies implementing agari-kit's public Agent protocol."""

from __future__ import annotations

from pathlib import Path

import torch
from agari_kit.agent import Context, ensure_legal
from agari_kit.engine import require_engine

from .features import ENCODING_VERSION, UPSTREAM_COMMIT, action_mask, policy_features
from .models.actor_critic import ActorCriticNetwork
from .models.q_network import QNetwork


class ModelAgent:
    def __init__(self, weights_path: str | Path, device: str = "cpu", torch_threads: int = 1):
        require_engine()
        if type(torch_threads) is not int or torch_threads < 1:
            raise ValueError("torch_threads must be positive")
        torch.set_num_threads(torch_threads)
        self.device = device
        self.metadata = torch.load(weights_path, map_location="cpu", weights_only=True)
        if (self.metadata.get("schema_version") != 1
                or self.metadata.get("encoding_version") != ENCODING_VERSION
                or self.metadata.get("upstream_commit") != UPSTREAM_COMMIT
                or self.metadata.get("engine_version") != "0.4.10"):
            raise ValueError("Incompatible model provenance or observation encoding")
        self.kind = self.metadata["kind"]
        classes = {"bc": QNetwork, "cql": QNetwork, "ppo": ActorCriticNetwork}
        if self.kind not in classes:
            raise ValueError("Only exported BC/CQL/PPO policies implement Agent")
        self.model = classes[self.kind](**self.metadata["model_config"])
        self.model.load_state_dict(self.metadata["state_dict"], strict=True)
        self.model.to(device).eval()

    def reset(self, context: Context) -> None:
        if context.rules.engine_version != "0.4.10" or context.rules.preset != "tenhou":
            raise ValueError("Policy requires pinned RiichiEnv / Tenhou rules")

    def act(self, observation, context: Context):
        if observation.player_id != context.player_id:
            raise ValueError("Observation belongs to another player")
        with torch.inference_mode():
            x = policy_features(observation).unsqueeze(0).to(self.device)
            output = self.model(x)
            scores = output[0] if self.kind == "ppo" else output
            mask = action_mask(observation).to(self.device)
            index = int(scores.squeeze(0).masked_fill(~mask, -torch.inf).argmax())
        return ensure_legal(observation.find_action(index), observation)
