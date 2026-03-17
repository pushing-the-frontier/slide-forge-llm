"""Render quality reward: successful rendering checks."""

from __future__ import annotations

from bs4 import BeautifulSoup


def render_quality_reward(completions: list, **kwargs) -> list[float]:
    """Score render success of generated slides.

    Expects kwargs: states (list of SlideForgeState).
    """
    states = kwargs.get("states", [])
    scores = []

    for i, completion in enumerate(completions):
        state = states[i] if i < len(states) else None
        if state is None:
            scores.append(0.0)
            continue

        brief = state.brief
        target = brief.num_slides if brief else 10
        total_html = len(state.slides_html)
        created = sum(1 for h in state.slides_html if h)
        rendered = sum(1 for p in state.slides_png if p)

        if created == 0:
            scores.append(0.0)
            continue

        score = 0.0

        # Coverage: how many of target slides were created
        score += 0.4 * min(created / target, 1.0)

        # Render success rate
        score += 0.3 * (rendered / max(created, 1))

        # HTML validity (parseable, has expected structure)
        valid = 0
        for html in state.slides_html:
            if not html:
                continue
            try:
                soup = BeautifulSoup(html, "html.parser")
                has_title = soup.select_one(".title") is not None
                has_sections = len(soup.select(".section")) > 0
                if has_title and has_sections:
                    valid += 1
            except Exception:
                pass
        score += 0.3 * (valid / max(created, 1))

        scores.append(score)

    return scores
