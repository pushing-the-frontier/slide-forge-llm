"""Content quality reward: topic relevance, factual grounding, narrative flow."""

from __future__ import annotations

from bs4 import BeautifulSoup


def content_quality_reward(completions: list, **kwargs) -> list[float]:
    """Score content quality of generated slides.

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

        topic_words = set(brief.topic.lower().split())
        score = 0.0

        # Topic relevance: check how many slides mention the topic
        relevant_slides = 0
        total_slides = 0
        for html in state.slides_html:
            if not html:
                continue
            total_slides += 1
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator=" ", strip=True).lower()
            text_words = set(text.split())
            if topic_words & text_words:
                relevant_slides += 1

        if total_slides > 0:
            score += 0.35 * (relevant_slides / total_slides)

        # Factual grounding: bonus if research was done
        if state.research_context:
            research_text = " ".join(
                str(r.get("results", "")) for r in state.research_context
            ).lower()
            grounding = 0
            for html in state.slides_html:
                if not html:
                    continue
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator=" ", strip=True).lower()
                # Check if slide content overlaps with research
                slide_words = set(text.split()) - {"the", "a", "an", "is", "are", "and", "or", "of", "to", "in"}
                research_words = set(research_text.split()) - {"the", "a", "an", "is", "are", "and", "or", "of", "to", "in"}
                overlap = len(slide_words & research_words)
                if overlap > 3:
                    grounding += 1
            if total_slides > 0:
                score += 0.25 * (grounding / total_slides)
        else:
            # No research penalty (unless confidence is high)
            if brief.confidence < 0.8:
                score += 0.0
            else:
                score += 0.15

        # Content uniqueness: slides should not repeat
        texts = []
        for html in state.slides_html:
            if not html:
                continue
            soup = BeautifulSoup(html, "html.parser")
            texts.append(soup.get_text(separator=" ", strip=True).lower())

        if len(texts) > 1:
            unique_ratio = len(set(texts)) / len(texts)
            score += 0.2 * unique_ratio
        elif texts:
            score += 0.2

        # Narrative flow: check title progression (simple heuristic)
        if state.outline:
            score += 0.2 * min(len(state.outline) / max(brief.num_slides, 1), 1.0)
        elif total_slides > 0:
            score += 0.1

        scores.append(min(score, 1.0))

    return scores
