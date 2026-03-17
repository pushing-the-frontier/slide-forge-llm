"""Meta tools: review_deck, finalize."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...models import SlideForgeState


def review_deck(state: SlideForgeState) -> tuple[str, bool]:
    """Review all slides for consistency and completeness."""
    brief = state.brief
    target = brief.num_slides if brief else 10
    created = sum(1 for h in state.slides_html if h)
    rendered = sum(1 for p in state.slides_png if p)

    issues = []
    if created < target:
        issues.append(f"Only {created}/{target} slides created.")
    if rendered < created:
        issues.append(f"{created - rendered} slides failed to render.")

    # Check word counts
    if brief:
        from bs4 import BeautifulSoup
        for i, html in enumerate(state.slides_html):
            if not html:
                continue
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            wc = len(text.split())
            target_wc = brief.word_count_per_slide
            if abs(wc - target_wc) > target_wc * 0.3:
                issues.append(f"Slide {i}: {wc} words (target ~{target_wc}).")

    if not issues:
        summary = f"Deck review: {created}/{target} slides, all checks passed."
    else:
        summary = f"Deck review: {created}/{target} slides.\nIssues:\n" + "\n".join(f"  - {i}" for i in issues)

    return summary, True


def finalize(state: SlideForgeState) -> tuple[str, bool]:
    """Mark the presentation as complete."""
    brief = state.brief
    target = brief.num_slides if brief else 10
    created = sum(1 for h in state.slides_html if h)

    if created == 0:
        return "Cannot finalize: no slides created.", False

    state.phase = "DONE"
    state.done = True
    return f"Presentation finalized with {created}/{target} slides.", True
