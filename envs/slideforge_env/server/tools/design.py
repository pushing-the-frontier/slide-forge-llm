"""Design tools: generate_slide, edit_slide, set_theme."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...models import SlideForgeState

from ..rendering.html_generator import generate_slide_html
from ..rendering.renderer import render_slide, png_to_base64


def generate_slide(
    state: SlideForgeState,
    slide_idx: int,
    title: str,
    sections: list[dict],
) -> tuple[str, bool]:
    """Generate a slide with the given title and sections.

    sections: list of {"heading": str, "body": str}
    """
    brief = state.brief
    theme = state.theme
    color_intensity = brief.colors if brief else 0.5
    target_slides = brief.num_slides if brief else 10

    html = generate_slide_html(
        title=title,
        sections=sections,
        theme_name=theme,
        color_intensity=color_intensity,
        slide_index=slide_idx,
        total_slides=target_slides,
    )

    # Expand lists if needed
    while len(state.slides_html) <= slide_idx:
        state.slides_html.append("")
        state.slides_png.append(b"")

    state.slides_html[slide_idx] = html

    # Try rendering
    png_bytes = render_slide(html)
    if png_bytes:
        state.slides_png[slide_idx] = png_bytes
        b64 = png_to_base64(png_bytes)
        msg = f"Slide {slide_idx} generated and rendered ({len(sections)} sections)."
    else:
        state.slides_png[slide_idx] = b""
        b64 = ""
        msg = f"Slide {slide_idx} generated (render unavailable, {len(sections)} sections)."

    if state.phase in ("RESEARCH", "PLAN"):
        state.phase = "GENERATE"

    return msg, True


def edit_slide(
    state: SlideForgeState,
    slide_idx: int,
    title: str | None = None,
    sections: list[dict] | None = None,
) -> tuple[str, bool]:
    """Edit an existing slide."""
    if slide_idx < 0 or slide_idx >= len(state.slides_html):
        return f"Error: slide {slide_idx} does not exist yet.", False

    if not state.slides_html[slide_idx]:
        return f"Error: slide {slide_idx} has no content to edit.", False

    brief = state.brief
    color_intensity = brief.colors if brief else 0.5
    target_slides = brief.num_slides if brief else 10

    # Parse existing slide to extract current values if partial edit
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(state.slides_html[slide_idx], "html.parser")

    if title is None:
        title_el = soup.select_one(".title")
        title = title_el.get_text() if title_el else f"Slide {slide_idx}"

    if sections is None:
        sections = []
        for sec_el in soup.select(".section"):
            h2 = sec_el.select_one("h2")
            p = sec_el.select_one("p")
            sections.append({
                "heading": h2.get_text() if h2 else "",
                "body": p.get_text() if p else "",
            })

    html = generate_slide_html(
        title=title,
        sections=sections,
        theme_name=state.theme,
        color_intensity=color_intensity,
        slide_index=slide_idx,
        total_slides=target_slides,
    )
    state.slides_html[slide_idx] = html

    png_bytes = render_slide(html)
    if png_bytes:
        state.slides_png[slide_idx] = png_bytes

    state.phase = "REFINE"
    return f"Slide {slide_idx} edited ({len(sections)} sections).", True


def set_theme(state: SlideForgeState, theme_name: str) -> tuple[str, bool]:
    """Set the presentation theme."""
    from ..rendering.themes import THEMES
    if theme_name not in THEMES:
        available = ", ".join(THEMES.keys())
        return f"Unknown theme '{theme_name}'. Available: {available}", False

    state.theme = theme_name
    return f"Theme set to '{theme_name}'.", True
