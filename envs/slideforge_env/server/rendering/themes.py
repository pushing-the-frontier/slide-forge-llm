"""Theme definitions with color intensity mapping.

colors parameter: 0.0 = grayscale, 1.0 = vivid.
"""

from __future__ import annotations


THEMES = {
    "default": {
        "name": "Default",
        "bg": (255, 255, 255),
        "text": (33, 33, 33),
        "accent": (41, 98, 255),
        "secondary": (100, 181, 246),
    },
    "dark": {
        "name": "Dark",
        "bg": (30, 30, 30),
        "text": (240, 240, 240),
        "accent": (0, 200, 83),
        "secondary": (76, 175, 80),
    },
    "corporate": {
        "name": "Corporate",
        "bg": (245, 245, 245),
        "text": (44, 62, 80),
        "accent": (52, 73, 94),
        "secondary": (149, 165, 166),
    },
    "creative": {
        "name": "Creative",
        "bg": (255, 253, 231),
        "text": (33, 33, 33),
        "accent": (255, 87, 34),
        "secondary": (255, 167, 38),
    },
    "tech": {
        "name": "Tech",
        "bg": (18, 18, 18),
        "text": (224, 224, 224),
        "accent": (0, 229, 255),
        "secondary": (29, 233, 182),
    },
}


def _lerp_color(gray: tuple[int, int, int], vivid: tuple[int, int, int], t: float) -> str:
    r = int(gray[0] + (vivid[0] - gray[0]) * t)
    g = int(gray[1] + (vivid[1] - gray[1]) * t)
    b = int(gray[2] + (vivid[2] - gray[2]) * t)
    return f"rgb({r},{g},{b})"


def _to_gray(color: tuple[int, int, int]) -> tuple[int, int, int]:
    lum = int(0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
    return (lum, lum, lum)


def resolve_theme(theme_name: str, color_intensity: float) -> dict[str, str]:
    """Resolve a theme with given color intensity (0=grayscale, 1=vivid)."""
    t = max(0.0, min(1.0, color_intensity))
    theme = THEMES.get(theme_name, THEMES["default"])

    return {
        "name": theme["name"],
        "bg": _lerp_color(_to_gray(theme["bg"]), theme["bg"], t),
        "text": _lerp_color(_to_gray(theme["text"]), theme["text"], t),
        "accent": _lerp_color(_to_gray(theme["accent"]), theme["accent"], t),
        "secondary": _lerp_color(_to_gray(theme["secondary"]), theme["secondary"], t),
    }
