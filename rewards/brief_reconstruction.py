"""Brief reconstruction reward: can Claude recover the original brief from slides?

Given the full slide deck, asks Claude to predict what the original brief was
(topic, audience, intended slide count). Compares the prediction against
the actual brief to score how faithfully the slides convey their purpose.

A high score means the slides are coherent, on-topic, and clearly targeted
at the intended audience.
"""

from __future__ import annotations

import hashlib
import json
import os

import boto3

_client = None
_score_cache: dict[str, float] = {}

BEDROCK_MODEL_ID = "us.anthropic.claude-opus-4-6-v1"

RECONSTRUCTION_PROMPT = """You are analyzing a slide deck presentation. Based ONLY on the slide content, predict what the original brief/requirements were.

Return a JSON object with:
{
  "topic": "The main topic or title of this presentation",
  "audience": "Who this presentation targets (e.g. board of directors, venture capitalists, executives, engineers, general)",
  "num_slides": <how many slides were intended>,
  "key_themes": ["theme1", "theme2", "theme3"]
}

Return ONLY the JSON object. No explanation."""


def _get_client():
    global _client
    if _client is None:
        region = os.environ.get("AWS_REGION", "us-east-1")
        _client = boto3.client("bedrock-runtime", region_name=region)
    return _client


def _hash_slides(slides_html: list[str]) -> str:
    content = "brief_recon:" + "||".join(h or "" for h in slides_html)
    return hashlib.md5(content.encode()).hexdigest()


def _normalize(text: str) -> set[str]:
    """Lowercase, split, strip common stop words."""
    stop = {"the", "a", "an", "is", "are", "and", "or", "of", "to", "in", "for", "on", "with", "-", "&"}
    return {w for w in text.lower().split() if w not in stop and len(w) > 1}


def _predict_brief(slides_html: list[str]) -> dict:
    """Ask Claude to reconstruct the brief from the slide deck."""
    client = _get_client()

    deck_text = ""
    for i, html in enumerate(slides_html):
        deck_text += f"\n--- Slide {i + 1} ---\n{html}\n"

    response = client.converse(
        modelId=BEDROCK_MODEL_ID,
        system=[{"text": RECONSTRUCTION_PROMPT}],
        messages=[{
            "role": "user",
            "content": [{"text": f"Analyze this {len(slides_html)}-slide deck:\n{deck_text}"}],
        }],
        inferenceConfig={"maxTokens": 512, "temperature": 0.0},
    )

    output = response["output"]["message"]
    text = "\n".join(block["text"] for block in output["content"] if "text" in block)

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return json.loads(text.strip())


def _score_reconstruction(predicted: dict, actual_brief) -> float:
    """Compare predicted brief against actual. Returns score in [0, 1]."""

    # --- Topic similarity (0.40) ---
    actual_topic_words = _normalize(actual_brief.topic)
    predicted_topic_words = _normalize(predicted.get("topic", ""))
    if actual_topic_words:
        topic_overlap = len(actual_topic_words & predicted_topic_words) / len(actual_topic_words)
    else:
        topic_overlap = 0.0
    topic_score = min(topic_overlap, 1.0)

    # --- Audience match (0.25) ---
    actual_audience = actual_brief.audience.lower().strip()
    predicted_audience = predicted.get("audience", "").lower().strip()
    if actual_audience == predicted_audience:
        audience_score = 1.0
    elif actual_audience in predicted_audience or predicted_audience in actual_audience:
        audience_score = 0.7
    else:
        actual_aud_words = _normalize(actual_audience)
        pred_aud_words = _normalize(predicted_audience)
        overlap = len(actual_aud_words & pred_aud_words)
        audience_score = 0.4 * min(overlap / max(len(actual_aud_words), 1), 1.0)

    # --- Slide count accuracy (0.15) ---
    actual_count = actual_brief.num_slides
    predicted_count = predicted.get("num_slides", 0)
    if isinstance(predicted_count, str) and predicted_count.isdigit():
        predicted_count = int(predicted_count)
    if actual_count > 0 and isinstance(predicted_count, (int, float)):
        ratio = min(predicted_count, actual_count) / max(predicted_count, actual_count)
        count_score = ratio
    else:
        count_score = 0.0

    # --- Theme coverage (0.20) ---
    predicted_themes = predicted.get("key_themes", [])
    if predicted_themes and actual_topic_words:
        theme_words = set()
        for theme in predicted_themes:
            theme_words |= _normalize(str(theme))
        theme_overlap = len(actual_topic_words & theme_words) / len(actual_topic_words)
        theme_score = min(theme_overlap * 1.5, 1.0)
    else:
        theme_score = 0.0

    return (0.40 * topic_score
            + 0.25 * audience_score
            + 0.15 * count_score
            + 0.20 * theme_score)


def brief_reconstruction_reward(completions: list, **kwargs) -> list[float]:
    """Score how well the slide deck reflects the original brief.

    Asks Claude to reconstruct the brief from slides, then compares
    against the actual brief on topic, audience, slide count, and themes.

    Expects kwargs: states (list of SlideForgeState).
    """
    states = kwargs.get("states", [])
    scores = []

    for i, completion in enumerate(completions):
        state = states[i] if i < len(states) else None
        if state is None or not state.slides_html or state.brief is None:
            scores.append(0.0)
            continue

        non_empty = [h for h in state.slides_html if h]
        if not non_empty:
            scores.append(0.0)
            continue

        cache_key = _hash_slides(non_empty)
        if cache_key in _score_cache:
            scores.append(_score_cache[cache_key])
            continue

        try:
            predicted = _predict_brief(non_empty)
            score = _score_reconstruction(predicted, state.brief)
            _score_cache[cache_key] = score
            scores.append(score)
        except Exception:
            scores.append(0.0)

    return scores
