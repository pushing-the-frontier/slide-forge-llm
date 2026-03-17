"""Claude-based aesthetic scoring for slide presentations.

Uses Claude via AWS Bedrock to evaluate visual design quality of generated
slides in two modes:
  - HTML scoring: evaluates the raw HTML/CSS code
  - Visual scoring: evaluates rendered PNG screenshots of each slide

Caches results by content hash to avoid redundant API calls during
per-step reward computation.
"""

from __future__ import annotations

import hashlib
import json
import os

import boto3

_client = None
_html_score_cache: dict[str, float] = {}
_visual_score_cache: dict[str, float] = {}

BEDROCK_MODEL_ID = "us.anthropic.claude-opus-4-6-v1"

HTML_SCORING_PROMPT = """You are an expert presentation designer scoring slide aesthetics.

Evaluate each slide's HTML for:
1. Layout & structure (0.25): Clear title/section hierarchy, logical organization
2. Content balance (0.25): Appropriate density — not too sparse, not overcrowded
3. Visual styling (0.25): Modern CSS (gradients, shadows, rounded corners), color harmony, typography
4. Professional polish (0.25): Executive-ready, consistent formatting, data presentation

Score each slide from 0.0 to 1.0.

Respond with ONLY a JSON object: {"scores": [0.85, 0.72, ...]}
No explanation needed."""

VISUAL_SCORING_PROMPT = """You are an expert presentation designer scoring the visual quality of rendered slides.

You will see PNG screenshots of presentation slides. Evaluate each slide for:
1. Visual design (0.25): Color harmony, contrast, visual balance, modern aesthetics
2. Layout & spacing (0.25): Proper use of whitespace, alignment, section organization
3. Typography (0.25): Font hierarchy, readability, text sizing, appropriate density
4. Professional polish (0.25): Executive-ready appearance, consistency, overall impression

Score each slide from 0.0 to 1.0.

Respond with ONLY a JSON object: {"scores": [0.85, 0.72, ...]}
No explanation needed."""


def _get_client():
    global _client
    if _client is None:
        region = os.environ.get("AWS_REGION", "us-east-1")
        _client = boto3.client("bedrock-runtime", region_name=region)
    return _client


def _hash_slides(slides_html: list[str], salt: str = "") -> str:
    content = salt + "||".join(h or "" for h in slides_html)
    return hashlib.md5(content.encode()).hexdigest()


def _call_claude_for_scores(prompt: str, user_text: str, num_slides: int) -> list[float]:
    """Generic helper: send text to Claude and parse per-slide scores."""
    client = _get_client()

    response = client.converse(
        modelId=BEDROCK_MODEL_ID,
        system=[{"text": prompt}],
        messages=[{"role": "user", "content": [{"text": user_text}]}],
        inferenceConfig={"maxTokens": 256, "temperature": 0.0},
    )

    output = response["output"]["message"]
    text = "\n".join(block["text"] for block in output["content"] if "text" in block)

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    result = json.loads(text.strip())
    scores = result.get("scores", [])

    while len(scores) < num_slides:
        scores.append(0.0)
    return [max(0.0, min(1.0, float(s))) for s in scores[:num_slides]]


# ---------------------------------------------------------------------------
# HTML scoring
# ---------------------------------------------------------------------------

def _score_html_with_claude(slides_html: list[str]) -> list[float]:
    """Send slide HTML to Claude and get per-slide aesthetic scores."""
    slides_text = ""
    for i, html in enumerate(slides_html):
        slides_text += f"\n--- Slide {i + 1} ---\n{html or '(empty)'}\n"

    user_msg = f"Score these {len(slides_html)} slides:\n{slides_text}"
    return _call_claude_for_scores(HTML_SCORING_PROMPT, user_msg, len(slides_html))


# ---------------------------------------------------------------------------
# Visual (PNG) scoring
# ---------------------------------------------------------------------------

def _get_slide_pngs(state) -> list[bytes]:
    """Collect PNG bytes for each non-empty slide.

    Uses pre-rendered PNGs from state.slides_png when available,
    falls back to rendering HTML via Playwright.
    """
    from envs.slideforge_env.server.rendering.renderer import render_slide

    pngs: list[bytes] = []
    for idx, html in enumerate(state.slides_html):
        if not html:
            continue
        png = None
        if idx < len(state.slides_png):
            candidate = state.slides_png[idx]
            if candidate and candidate != b"":
                png = candidate
        if png is None:
            png = render_slide(html) or b""
        if png and png != b"":
            pngs.append(png)
    return pngs


def _score_visual_with_claude(png_list: list[bytes]) -> list[float]:
    """Send rendered PNG screenshots to Claude and get per-slide scores."""
    client = _get_client()

    content_blocks: list[dict] = []
    for i, png_bytes in enumerate(png_list):
        content_blocks.append({"text": f"Slide {i + 1}:"})
        content_blocks.append({
            "image": {
                "format": "png",
                "source": {"bytes": png_bytes},
            }
        })
    content_blocks.append({"text": f"Score these {len(png_list)} slides."})

    response = client.converse(
        modelId=BEDROCK_MODEL_ID,
        system=[{"text": VISUAL_SCORING_PROMPT}],
        messages=[{"role": "user", "content": content_blocks}],
        inferenceConfig={"maxTokens": 256, "temperature": 0.0},
    )

    output = response["output"]["message"]
    text = "\n".join(block["text"] for block in output["content"] if "text" in block)

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    result = json.loads(text.strip())
    scores = result.get("scores", [])

    while len(scores) < len(png_list):
        scores.append(0.0)
    return [max(0.0, min(1.0, float(s))) for s in scores[:len(png_list)]]


# ---------------------------------------------------------------------------
# Public reward functions
# ---------------------------------------------------------------------------

def claude_aesthetic_html_reward(completions: list, **kwargs) -> list[float]:
    """Score HTML slide aesthetics using Claude via Bedrock.

    Expects kwargs: states (list of SlideForgeState).
    """
    states = kwargs.get("states", [])
    scores = []

    for i, completion in enumerate(completions):
        state = states[i] if i < len(states) else None
        if state is None or not state.slides_html:
            scores.append(0.0)
            continue

        non_empty = [h for h in state.slides_html if h]
        if not non_empty:
            scores.append(0.0)
            continue

        cache_key = _hash_slides(non_empty, salt="html:")
        if cache_key in _html_score_cache:
            scores.append(_html_score_cache[cache_key])
            continue

        try:
            slide_scores = _score_html_with_claude(non_empty)
            avg = sum(slide_scores) / len(slide_scores)
            _html_score_cache[cache_key] = avg
            scores.append(avg)
        except Exception:
            scores.append(0.0)

    return scores


def claude_aesthetic_visual_reward(completions: list, **kwargs) -> list[float]:
    """Score rendered slide screenshots using Claude via Bedrock.

    Uses pre-rendered PNGs from state.slides_png (Playwright screenshots),
    falling back to on-demand rendering when PNGs are missing.

    Expects kwargs: states (list of SlideForgeState).
    """
    states = kwargs.get("states", [])
    scores = []

    for i, completion in enumerate(completions):
        state = states[i] if i < len(states) else None
        if state is None or not state.slides_html:
            scores.append(0.0)
            continue

        non_empty = [h for h in state.slides_html if h]
        if not non_empty:
            scores.append(0.0)
            continue

        cache_key = _hash_slides(non_empty, salt="visual:")
        if cache_key in _visual_score_cache:
            scores.append(_visual_score_cache[cache_key])
            continue

        try:
            pngs = _get_slide_pngs(state)
            if not pngs:
                scores.append(0.0)
                continue
            slide_scores = _score_visual_with_claude(pngs)
            avg = sum(slide_scores) / len(slide_scores)
            _visual_score_cache[cache_key] = avg
            scores.append(avg)
        except Exception:
            scores.append(0.0)

    return scores
