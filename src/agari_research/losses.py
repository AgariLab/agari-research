"""Losses adapted from RiichiEnv's Apache-2.0 CQL and PPO trainers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bc_loss(q_values, actions, masks):
    """Hard behavior labels; masked-out native actions never receive gradients."""
    if not torch.all(masks.gather(1, actions[:, None])):
        raise ValueError("BC actions must be legal")
    logits = q_values.masked_fill(~masks, -torch.inf)
    loss = F.cross_entropy(logits, actions)
    return loss, {"cross_entropy": loss,
                  "accuracy": (logits.argmax(1) == actions).float().mean()}


def cql_loss(q_values, aux_logits, actions, masks, targets, ranks, *, alpha=5.0, aux_weight=0.2):
    if not torch.all(masks.gather(1, actions[:, None])):
        raise ValueError("CQL actions must be legal")
    chosen = q_values.gather(1, actions[:, None]).squeeze(1)
    conservative = (q_values.masked_fill(~masks, -1e9).logsumexp(dim=1) - chosen).mean()
    td = F.mse_loss(chosen, targets)
    auxiliary = F.cross_entropy(aux_logits, ranks)
    loss = td + alpha * conservative + aux_weight * auxiliary
    return loss, {"td_loss": td, "conservative_loss": conservative, "aux_loss": auxiliary}


def gae(values: list[float], terminal_reward: float, gamma: float, lam: float):
    """A complete kyoku is a terminal trajectory, including repeated dealer hands."""
    advantages, returns = [0.0] * len(values), [0.0] * len(values)
    advantage = next_value = 0.0
    for index in range(len(values) - 1, -1, -1):
        reward = terminal_reward if index == len(values) - 1 else 0.0
        delta = reward + gamma * next_value - values[index]
        advantage = delta + gamma * lam * advantage
        advantages[index], returns[index] = advantage, advantage + values[index]
        next_value = values[index]
    return advantages, returns


def ppo_loss(logits, values, reference_logits, masks, actions, old_log_probs,
             advantages, returns, *, clip, value_coef, entropy_coef, kl_coef):
    logits = logits.masked_fill(~masks, -1e9)
    reference_logits = reference_logits.masked_fill(~masks, -1e9)
    distribution = torch.distributions.Categorical(logits=logits)
    ratio = (distribution.log_prob(actions) - old_log_probs).exp()
    policy = -torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages).mean()
    value = F.mse_loss(values, returns)
    entropy = distribution.entropy().mean()
    log_probs = F.log_softmax(logits, dim=-1)
    kl = (log_probs.exp() * (log_probs - F.log_softmax(reference_logits, dim=-1))).sum(dim=-1).mean()
    total = policy + value_coef * value - entropy_coef * entropy + kl_coef * kl
    return total, {"policy_loss": policy, "value_loss": value, "entropy": entropy, "kl": kl}
