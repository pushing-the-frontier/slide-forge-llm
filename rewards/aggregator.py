"""Weighted reward aggregation."""

from __future__ import annotations

from .code_rules import code_rules_reward
from .render_quality import render_quality_reward
from .claude_aesthetic import claude_aesthetic_html_reward, claude_aesthetic_visual_reward
from .content_quality import content_quality_reward
from .brief_reconstruction import brief_reconstruction_reward

DEFAULT_WEIGHTS = {
    "code_rules": 1.0,
    "render_quality": 2.0,
    "claude_aesthetic_html": 1.5,
    "claude_aesthetic_visual": 1.5,
    "content_quality": 2.0,
    "brief_reconstruction": 2.0,
}

REWARD_FUNCTIONS = {
    "code_rules": code_rules_reward,
    "render_quality": render_quality_reward,
    "claude_aesthetic_html": claude_aesthetic_html_reward,
    "claude_aesthetic_visual": claude_aesthetic_visual_reward,
    "content_quality": content_quality_reward,
    "brief_reconstruction": brief_reconstruction_reward,
}


def aggregate_rewards(
    completions: list,
    weights: dict[str, float] | None = None,
    **kwargs,
) -> list[float]:
    """Compute weighted aggregate reward across all components."""
    w = weights or DEFAULT_WEIGHTS
    total_weight = sum(w.values())
    if total_weight == 0:
        return [0.0] * len(completions)

    component_scores = {}
    for name, fn in REWARD_FUNCTIONS.items():
        if name in w and w[name] > 0:
            try:
                component_scores[name] = fn(completions, **kwargs)
            except Exception:
                component_scores[name] = [0.0] * len(completions)

    aggregated = []
    for i in range(len(completions)):
        total = 0.0
        for name, scores in component_scores.items():
            if i < len(scores):
                total += w.get(name, 0.0) * scores[i]
        aggregated.append(total / total_weight)

    return aggregated


def compute_reward_details(
    completions: list,
    weights: dict[str, float] | None = None,
    **kwargs,
) -> list[dict]:
    """Compute per-component reward breakdown."""
    w = weights or DEFAULT_WEIGHTS

    component_scores = {}
    for name, fn in REWARD_FUNCTIONS.items():
        if name in w and w[name] > 0:
            try:
                component_scores[name] = fn(completions, **kwargs)
            except Exception:
                component_scores[name] = [0.0] * len(completions)

    details = []
    total_weight = sum(w.values())
    for i in range(len(completions)):
        d = {}
        total = 0.0
        for name, scores in component_scores.items():
            s = scores[i] if i < len(scores) else 0.0
            d[name] = s
            total += w.get(name, 0.0) * s
        d["aggregate"] = total / total_weight if total_weight > 0 else 0.0
        details.append(d)

    return details
