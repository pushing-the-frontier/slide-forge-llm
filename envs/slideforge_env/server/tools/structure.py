"""Structural editing tools: get_slide_content, delete_slide, reorder_slides, duplicate_slide, insert_slide."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...models import SlideForgeState


def get_slide_content(state: SlideForgeState, slide_idx: int) -> tuple[str, bool]:
    """Extract and return the text content of a slide for inspection."""
    if slide_idx < 0 or slide_idx >= len(state.slides_html):
        return f"Error: slide {slide_idx} does not exist (0-{len(state.slides_html) - 1}).", False

    html = state.slides_html[slide_idx]
    if not html:
        return f"Slide {slide_idx} exists but has no content.", False

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    title_el = soup.select_one(".title")
    title = title_el.get_text(strip=True) if title_el else "(no title)"

    sections = []
    for sec_el in soup.select(".section"):
        h2 = sec_el.select_one("h2")
        p = sec_el.select_one("p")
        heading = h2.get_text(strip=True) if h2 else ""
        body = p.get_text(strip=True) if p else ""
        sections.append({"heading": heading, "body": body})

    word_count = len(soup.get_text(separator=" ", strip=True).split())
    lines = [
        f"Slide {slide_idx}: \"{title}\" ({len(sections)} sections, ~{word_count} words)",
    ]
    for i, sec in enumerate(sections):
        lines.append(f"  Section {i}: [{sec['heading']}] {sec['body'][:120]}{'...' if len(sec['body']) > 120 else ''}")

    return "\n".join(lines), True


def delete_slide(state: SlideForgeState, slide_idx: int) -> tuple[str, bool]:
    """Remove a slide from the deck."""
    if slide_idx < 0 or slide_idx >= len(state.slides_html):
        return f"Error: slide {slide_idx} does not exist (0-{len(state.slides_html) - 1}).", False

    if len(state.slides_html) <= 1:
        return "Error: cannot delete the last remaining slide.", False

    state.slides_html.pop(slide_idx)
    state.slides_png.pop(slide_idx)

    if state.outline and slide_idx < len(state.outline):
        state.outline.pop(slide_idx)
        for i, entry in enumerate(state.outline):
            entry["slide_index"] = i

    state.phase = "REFINE"
    return f"Deleted slide {slide_idx}. Deck now has {len(state.slides_html)} slides.", True


def reorder_slides(state: SlideForgeState, from_idx: int, to_idx: int) -> tuple[str, bool]:
    """Move a slide from one position to another."""
    n = len(state.slides_html)
    if from_idx < 0 or from_idx >= n:
        return f"Error: from_idx {from_idx} out of range (0-{n - 1}).", False
    if to_idx < 0 or to_idx >= n:
        return f"Error: to_idx {to_idx} out of range (0-{n - 1}).", False
    if from_idx == to_idx:
        return f"Slide {from_idx} is already at position {to_idx}.", True

    html = state.slides_html.pop(from_idx)
    png = state.slides_png.pop(from_idx)
    state.slides_html.insert(to_idx, html)
    state.slides_png.insert(to_idx, png)

    if state.outline and from_idx < len(state.outline):
        entry = state.outline.pop(from_idx)
        state.outline.insert(min(to_idx, len(state.outline)), entry)
        for i, e in enumerate(state.outline):
            e["slide_index"] = i

    state.phase = "REFINE"
    return f"Moved slide from position {from_idx} to {to_idx}.", True


def duplicate_slide(state: SlideForgeState, slide_idx: int) -> tuple[str, bool]:
    """Duplicate a slide, inserting the copy right after the original."""
    if slide_idx < 0 or slide_idx >= len(state.slides_html):
        return f"Error: slide {slide_idx} does not exist (0-{len(state.slides_html) - 1}).", False

    html_copy = state.slides_html[slide_idx]
    png_copy = state.slides_png[slide_idx] if slide_idx < len(state.slides_png) else b""

    insert_at = slide_idx + 1
    state.slides_html.insert(insert_at, html_copy)
    state.slides_png.insert(insert_at, png_copy)

    if state.outline and slide_idx < len(state.outline):
        import copy
        entry_copy = copy.deepcopy(state.outline[slide_idx])
        state.outline.insert(insert_at, entry_copy)
        for i, e in enumerate(state.outline):
            e["slide_index"] = i

    state.phase = "REFINE"
    return f"Duplicated slide {slide_idx} at position {insert_at}. Deck now has {len(state.slides_html)} slides.", True


def insert_slide(
    state: SlideForgeState,
    slide_idx: int,
    title: str,
    sections: list[dict],
) -> tuple[str, bool]:
    """Insert a new slide at the given position, shifting subsequent slides down."""
    n = len(state.slides_html)
    if slide_idx < 0 or slide_idx > n:
        return f"Error: slide_idx {slide_idx} out of range (0-{n}).", False

    from ..rendering.html_generator import generate_slide_html
    from ..rendering.renderer import render_slide

    brief = state.brief
    color_intensity = brief.colors if brief else 0.5
    total_slides = max(n + 1, brief.num_slides if brief else 10)

    html = generate_slide_html(
        title=title,
        sections=sections,
        theme_name=state.theme,
        color_intensity=color_intensity,
        slide_index=slide_idx,
        total_slides=total_slides,
    )

    state.slides_html.insert(slide_idx, html)
    png_bytes = render_slide(html)
    state.slides_png.insert(slide_idx, png_bytes or b"")

    state.phase = "REFINE"
    return f"Inserted new slide at position {slide_idx}. Deck now has {len(state.slides_html)} slides.", True
