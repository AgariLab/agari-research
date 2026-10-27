"""Fixed upstream v1 policy encoding, with separate training-only reward features."""

from __future__ import annotations

import numpy as np
import torch

ENCODING_VERSION = "riichienv-0.4.10-feat-v1-74x34-actions82"
UPSTREAM_COMMIT = "479c1faeb33d082965eef8198f63261a79c0fce3"


def policy_features(observation) -> torch.Tensor:
    raw = observation.encode()
    if len(raw) != 74 * 34 * 4:
        raise ValueError("Unexpected pinned-engine feature shape")
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).reshape(74, 34).copy())


def action_mask(observation) -> torch.Tensor:
    mask = np.frombuffer(observation.mask(), dtype=np.uint8).copy()
    if mask.shape != (82,) or not mask.any():
        raise ValueError("Unexpected or empty native action mask")
    return torch.from_numpy(mask.astype(bool))


def grp_features(features: dict, player: int) -> torch.Tensor:
    """Outcome features are for GRP / training rewards only, never policy inputs."""
    scores = list(features["round_initial_scores"]) + list(features["round_end_scores"])
    values = [score / 25000.0 for score in scores]
    values.extend(delta / 12000.0 for delta in features["round_delta_scores"])
    values.extend([features["chang"] / 3.0, features["ju"] / 3.0,
                   features["ben"] / 4.0, features["liqibang"] / 4.0])
    values.extend(float(seat == player) for seat in range(4))
    if len(values) != 20:
        raise ValueError("Invalid four-player GRP feature vector")
    return torch.tensor(values, dtype=torch.float32)


def rank_of(scores: list[int], player: int) -> int:
    return sorted(range(4), key=lambda seat: (-scores[seat], seat)).index(player)


def predicted_rewards(model, features: dict, weights: list[float], device: str) -> list[float]:
    with torch.inference_mode():
        x = torch.stack([grp_features(features, seat) for seat in range(4)]).to(device)
        points = torch.softmax(model(x), dim=-1) @ torch.tensor(weights, device=device)
        return (points - sum(weights) / 4).cpu().tolist()
