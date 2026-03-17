"""Generate HTML for individual slides."""

from __future__ import annotations

from .themes import resolve_theme


def generate_slide_html(
    title: str,
    sections: list[dict],
    theme_name: str = "default",
    color_intensity: float = 0.5,
    slide_index: int = 0,
    total_slides: int = 1,
) -> str:
    """Generate a single slide as a self-contained HTML string.

    Each section dict should have: {"heading": str, "body": str}
    """
    colors = resolve_theme(theme_name, color_intensity)

    sections_html = ""
    for i, sec in enumerate(sections):
        heading = sec.get("heading", "")
        body = sec.get("body", "")
        # Add visual variety with alternating section styles
        icon = _get_section_icon(i, heading)
        sections_html += f"""
        <div class="section">
            <div class="section-header">
                <span class="section-icon">{icon}</span>
                <h2>{heading}</h2>
            </div>
            <p>{body}</p>
        </div>"""

    # Determine layout based on section count
    layout_class = "layout-" + ("two" if len(sections) == 2 else "three" if len(sections) >= 3 else "one")

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    
    body {{
        width: 1280px;
        height: 720px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: {colors['bg']};
        color: {colors['text']};
        display: flex;
        flex-direction: column;
        padding: 56px 72px;
        overflow: hidden;
        position: relative;
    }}
    
    /* Decorative background elements */
    body::before {{
        content: '';
        position: absolute;
        top: -100px;
        right: -100px;
        width: 400px;
        height: 400px;
        background: {colors['accent']};
        opacity: 0.03;
        border-radius: 50%;
    }}
    
    body::after {{
        content: '';
        position: absolute;
        bottom: -50px;
        left: -50px;
        width: 200px;
        height: 200px;
        background: {colors['secondary']};
        opacity: 0.05;
        border-radius: 50%;
    }}
    
    .slide-number {{
        position: absolute;
        top: 24px;
        right: 32px;
        font-size: 13px;
        font-weight: 500;
        color: {colors['secondary']};
        letter-spacing: 0.5px;
    }}
    
    .title {{
        font-size: 42px;
        font-weight: 700;
        color: {colors['accent']};
        margin-bottom: 12px;
        line-height: 1.2;
        letter-spacing: -0.5px;
    }}
    
    .title-underline {{
        width: 80px;
        height: 4px;
        background: linear-gradient(90deg, {colors['accent']}, {colors['secondary']});
        border-radius: 2px;
        margin-bottom: 40px;
    }}
    
    .sections {{
        display: flex;
        gap: 32px;
        flex: 1;
        align-items: stretch;
    }}
    
    .sections.layout-one {{
        flex-direction: column;
    }}
    
    .sections.layout-two,
    .sections.layout-three {{
        flex-direction: row;
    }}
    
    .section {{
        flex: 1;
        background: rgba(255, 255, 255, 0.6);
        border-radius: 16px;
        padding: 28px;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease;
    }}
    
    .section:hover {{
        transform: translateY(-2px);
    }}
    
    .section-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
    }}
    
    .section-icon {{
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, {colors['accent']}, {colors['secondary']});
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        flex-shrink: 0;
    }}
    
    .section h2 {{
        font-size: 18px;
        font-weight: 600;
        color: {colors['accent']};
        line-height: 1.3;
    }}
    
    .section p {{
        font-size: 15px;
        line-height: 1.7;
        color: {colors['text']};
        opacity: 0.85;
        flex: 1;
    }}
    
    .footer {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 24px;
        padding-top: 16px;
        border-top: 1px solid rgba(0, 0, 0, 0.06);
    }}
    
    .footer-brand {{
        font-size: 12px;
        font-weight: 500;
        color: {colors['secondary']};
        letter-spacing: 1px;
        text-transform: uppercase;
    }}
    
    .footer-progress {{
        display: flex;
        gap: 6px;
    }}
    
    .progress-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: {colors['secondary']};
        opacity: 0.3;
    }}
    
    .progress-dot.active {{
        background: {colors['accent']};
        opacity: 1;
    }}
</style>
</head>
<body>
    <div class="slide-number">{slide_index + 1:02d} / {total_slides:02d}</div>
    <div class="title">{title}</div>
    <div class="title-underline"></div>
    <div class="sections {layout_class}">{sections_html}
    </div>
    <div class="footer">
        <div class="footer-brand">SlideForge</div>
        <div class="footer-progress">
            {_generate_progress_dots(slide_index, total_slides, colors)}
        </div>
    </div>
</body>
</html>"""


def _get_section_icon(index: int, heading: str) -> str:
    """Get an appropriate icon for the section based on content."""
    heading_lower = heading.lower()
    
    # Content-based icons
    if any(w in heading_lower for w in ['growth', 'revenue', 'roi', 'value', 'profit']):
        return '📈'
    if any(w in heading_lower for w in ['risk', 'security', 'threat', 'privacy']):
        return '🛡️'
    if any(w in heading_lower for w in ['customer', 'user', 'experience']):
        return '👥'
    if any(w in heading_lower for w in ['data', 'analytics', 'insight']):
        return '📊'
    if any(w in heading_lower for w in ['strategy', 'roadmap', 'plan']):
        return '🎯'
    if any(w in heading_lower for w in ['innovation', 'future', 'trend']):
        return '💡'
    if any(w in heading_lower for w in ['automation', 'efficiency', 'process']):
        return '⚙️'
    if any(w in heading_lower for w in ['team', 'talent', 'culture']):
        return '🤝'
    if any(w in heading_lower for w in ['market', 'competitive', 'industry']):
        return '🏆'
    if any(w in heading_lower for w in ['technology', 'tech', 'ai', 'ml']):
        return '🚀'
    
    # Fallback based on position
    icons = ['✨', '🔷', '◆', '●', '▸']
    return icons[index % len(icons)]


def _generate_progress_dots(current: int, total: int, colors: dict) -> str:
    """Generate progress indicator dots."""
    dots = []
    for i in range(total):
        active = "active" if i == current else ""
        dots.append(f'<div class="progress-dot {active}"></div>')
    return "\n            ".join(dots)
