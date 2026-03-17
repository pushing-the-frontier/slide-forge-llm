"""Content tools: create_outline, revise_outline."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...models import SlideForgeState


def create_outline(state: SlideForgeState, sections: list[dict]) -> tuple[str, bool]:
    """Create a slide outline.

    sections: list of {"title": str, "bullet_points": list[str]}
    """
    if not sections:
        return "Error: sections list cannot be empty.", False

    brief = state.brief
    target = brief.num_slides if brief else 10

    state.outline = []
    for i, sec in enumerate(sections):
        title = sec.get("title", f"Slide {i + 1}")
        bullets = sec.get("bullet_points", [])
        sps = brief.sections_per_slide if brief else 3
        if brief and brief.slide_sections and str(i) in brief.slide_sections:
            sps = brief.slide_sections[str(i)]
        state.outline.append({
            "slide_index": i,
            "title": title,
            "bullet_points": bullets[:sps],
            "sections_per_slide": sps,
        })

    state.phase = "PLAN"
    summary_lines = [f"Outline created with {len(state.outline)} slides (target: {target}):"]
    for entry in state.outline:
        summary_lines.append(f"  [{entry['slide_index']}] {entry['title']} ({len(entry['bullet_points'])} bullets)")
    return "\n".join(summary_lines), True


def revise_outline(
    state: SlideForgeState,
    slide_index: int,
    title: str | None = None,
    bullet_points: list[str] | None = None,
) -> tuple[str, bool]:
    """Revise a specific slide in the outline."""
    if slide_index < 0 or slide_index >= len(state.outline):
        return f"Error: slide_index {slide_index} out of range (0-{len(state.outline) - 1}).", False

    entry = state.outline[slide_index]
    if title is not None:
        entry["title"] = title
    if bullet_points is not None:
        sps = entry.get("sections_per_slide", 3)
        entry["bullet_points"] = bullet_points[:sps]

    return f"Revised slide {slide_index}: {entry['title']} ({len(entry['bullet_points'])} bullets)", True
