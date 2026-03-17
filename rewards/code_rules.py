"""Structural validation reward: word count, sections, required elements."""

from __future__ import annotations

from bs4 import BeautifulSoup


def code_rules_reward(completions: list, **kwargs) -> list[float]:
    """Score structural adherence of generated slides.

    Expects kwargs: states (list of SlideForgeState).
    """
    states = kwargs.get("states", [])
    scores = []

    for i, completion in enumerate(completions):
        state = states[i] if i < len(states) else None
        if state is None or not state.slides_html:
            scores.append(0.0)
            continue

        brief = state.brief
        if brief is None:
            scores.append(0.0)
            continue

        slide_scores = []
        for idx, html in enumerate(state.slides_html):
            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")
            slide_score = 0.0

            # Title present
            title_el = soup.select_one(".title")
            if title_el and title_el.get_text(strip=True):
                slide_score += 0.25

            # Section count adherence
            section_els = soup.select(".section")
            target_sections = brief.sections_per_slide
            if brief.slide_sections and str(idx) in brief.slide_sections:
                target_sections = brief.slide_sections[str(idx)]
            if len(section_els) == target_sections:
                slide_score += 0.25
            elif section_els:
                slide_score += 0.1

            # Word count adherence
            text = soup.get_text(separator=" ", strip=True)
            word_count = len(text.split())
            target_wc = brief.word_count_per_slide
            if target_wc > 0:
                ratio = min(word_count, target_wc) / max(word_count, target_wc)
                slide_score += 0.25 * ratio

            # Content in sections (not empty)
            non_empty = sum(1 for s in section_els if s.get_text(strip=True))
            if section_els:
                slide_score += 0.25 * (non_empty / len(section_els))

            slide_scores.append(slide_score)

        scores.append(sum(slide_scores) / max(len(slide_scores), 1))

    return scores
