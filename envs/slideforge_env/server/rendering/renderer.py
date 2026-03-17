"""Playwright-based screenshot renderer for HTML slides with monitoring."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_BROWSER = None
_PLAYWRIGHT = None


@dataclass
class RenderStats:
    """Tracks renderer health across the process lifetime."""
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    total_render_ms: float = 0.0
    last_error: str = ""
    backend: str = "none"

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0

    @property
    def avg_render_ms(self) -> float:
        return self.total_render_ms / self.successes if self.successes else 0.0

    def summary(self) -> str:
        return (
            f"Renderer[{self.backend}]: {self.successes}/{self.attempts} "
            f"({self.success_rate:.0%}), avg {self.avg_render_ms:.0f}ms"
            + (f", last_error={self.last_error}" if self.last_error else "")
        )


stats = RenderStats()


# ---------------------------------------------------------------------------
# Playwright backend
# ---------------------------------------------------------------------------

async def _get_browser():
    global _BROWSER, _PLAYWRIGHT
    if _BROWSER is None:
        from playwright.async_api import async_playwright
        _PLAYWRIGHT = await async_playwright().start()
        _BROWSER = await _PLAYWRIGHT.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-gpu"],
        )
        stats.backend = "playwright"
        logger.info("Playwright browser launched")
    return _BROWSER


async def render_slide_async(html: str, width: int = 1280, height: int = 720) -> Optional[bytes]:
    """Render HTML to PNG bytes using Playwright."""
    timeout = int(os.environ.get("SLIDEFORGE_RENDER_TIMEOUT", "30")) * 1000
    page = None
    try:
        browser = await _get_browser()
        page = await browser.new_page(viewport={"width": width, "height": height})
        await page.set_content(html, wait_until="networkidle", timeout=timeout)
        png_bytes = await page.screenshot(type="png")
        return png_bytes
    except Exception as e:
        logger.warning("Playwright render failed: %s", e)
        raise
    finally:
        if page:
            try:
                await page.close()
            except Exception:
                pass


def _run_async(coro):
    """Run an async coroutine from sync context, handling event loop edge cases."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Pillow fallback (no browser needed)
# ---------------------------------------------------------------------------

def _render_with_pillow(html: str, width: int = 1280, height: int = 720) -> Optional[bytes]:
    """Basic fallback: render slide structure using Pillow.

    Produces a simpler image (no CSS styling) but guarantees we always
    get a PNG even when Playwright/Chromium are unavailable.
    """
    import io
    from PIL import Image, ImageDraw, ImageFont
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    title_el = soup.select_one(".title")
    title = title_el.get_text(strip=True) if title_el else "Slide"

    sections = []
    for sec in soup.select(".section"):
        h2 = sec.select_one("h2")
        p = sec.select_one("p")
        sections.append({
            "heading": h2.get_text(strip=True) if h2 else "",
            "body": p.get_text(strip=True) if p else sec.get_text(strip=True),
        })

    bg_color = (30, 30, 30)
    text_color = (230, 230, 230)
    accent_color = (80, 140, 255)
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        heading_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        title_font = ImageFont.load_default()
        heading_font = title_font
        body_font = title_font

    margin = 50
    y = 40

    draw.text((margin, y), title, fill=accent_color, font=title_font)
    y += 50

    draw.line([(margin, y), (margin + 100, y)], fill=accent_color, width=3)
    y += 25

    if sections:
        n = len(sections)
        col_width = (width - 2 * margin - (n - 1) * 20) // max(n, 1)

        for i, sec in enumerate(sections):
            col_x = margin + i * (col_width + 20)
            cy = y

            draw.text((col_x, cy), sec["heading"], fill=accent_color, font=heading_font)
            cy += 28

            body = sec["body"]
            chars_per_line = max(col_width // 9, 20)
            for line_start in range(0, len(body), chars_per_line):
                line = body[line_start:line_start + chars_per_line]
                if cy > height - 40:
                    break
                draw.text((col_x, cy), line, fill=text_color, font=body_font)
                cy += 20

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_slide(html: str, width: int = 1280, height: int = 720) -> Optional[bytes]:
    """Render HTML slide to PNG with monitoring and fallback.

    Tries Playwright first, falls back to Pillow-based rendering.
    All attempts and failures are tracked in the module-level `stats` object.
    """
    stats.attempts += 1
    t0 = time.perf_counter()

    # Primary: Playwright
    try:
        result = _run_async(render_slide_async(html, width, height))
        if result:
            elapsed = (time.perf_counter() - t0) * 1000
            stats.successes += 1
            stats.total_render_ms += elapsed
            if stats.successes == 1 or stats.successes % 50 == 0:
                logger.info(stats.summary())
            return result
    except Exception as e:
        stats.last_error = f"playwright: {e}"
        logger.debug("Playwright unavailable, trying Pillow fallback")

    # Fallback: Pillow
    try:
        result = _render_with_pillow(html, width, height)
        if result:
            elapsed = (time.perf_counter() - t0) * 1000
            stats.successes += 1
            stats.total_render_ms += elapsed
            if stats.backend != "pillow":
                stats.backend = "pillow"
                logger.info("Using Pillow fallback renderer")
            return result
    except Exception as e:
        stats.last_error = f"pillow: {e}"
        logger.warning("Pillow fallback failed: %s", e)

    stats.failures += 1
    if stats.failures <= 3 or stats.failures % 20 == 0:
        logger.error("Render failed (%s). %s", stats.last_error, stats.summary())
    return None


def png_to_base64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("utf-8")


def get_render_stats() -> RenderStats:
    """Access renderer health stats from outside the module."""
    return stats
