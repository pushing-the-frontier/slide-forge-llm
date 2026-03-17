"""DINOv2-based aesthetic scoring for slide images.

Uses feature diversity and spatial coherence from DINOv2 embeddings
to evaluate visual quality without reference images.

Falls back to HTML heuristics when PNG rendering is unavailable.
"""

from __future__ import annotations

import io
from typing import Optional

_model = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is None:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        _processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        _model = AutoModel.from_pretrained("facebook/dinov2-small")
        _model.eval()
        if torch.cuda.is_available():
            _model = _model.cuda()
    return _model, _processor


def score_slide_image(png_bytes: bytes) -> float:
    """Score a single slide image using DINOv2 features.

    Evaluates:
    - Feature diversity (spread of patch tokens -> layout variety)
    - Spatial coherence (smoothness of neighboring patch features)
    - Overall activation magnitude (visual richness)

    Returns a score in [0, 1].
    """
    import torch
    from PIL import Image

    model, processor = _load_model()
    device = next(model.parameters()).device

    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # skip CLS token

    # Feature diversity: std across patches (higher = more varied layout)
    diversity = patch_tokens.std(dim=1).mean().item()
    diversity_score = min(diversity / 0.5, 1.0)

    # Spatial coherence: mean cosine sim between adjacent patches
    t = patch_tokens.squeeze(0)
    if t.shape[0] > 1:
        cos_sim = torch.nn.functional.cosine_similarity(t[:-1], t[1:], dim=-1)
        coherence = cos_sim.mean().item()
        coherence_score = max(0.0, coherence)
    else:
        coherence_score = 0.5

    # Activation magnitude: overall feature richness
    magnitude = patch_tokens.norm(dim=-1).mean().item()
    magnitude_score = min(magnitude / 20.0, 1.0)

    # Weighted combination
    score = 0.4 * diversity_score + 0.35 * coherence_score + 0.25 * magnitude_score
    return max(0.0, min(1.0, score))


def score_slide_html(html: str) -> float:
    """Score a slide based on HTML structure and content heuristics.
    
    Fallback when PNG rendering is unavailable.
    Evaluates:
    - Structure quality (proper sections, headings)
    - Content richness (text length, variety)
    - Visual elements (icons, styling indicators)
    
    Returns a score in [0, 1].
    """
    from bs4 import BeautifulSoup
    
    if not html:
        return 0.0
    
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return 0.0
    
    score = 0.0
    
    # Structure quality (0.35)
    has_title = soup.select_one(".title") is not None
    sections = soup.select(".section")
    has_sections = len(sections) > 0
    has_footer = soup.select_one(".footer") is not None
    
    structure_score = 0.0
    if has_title:
        structure_score += 0.4
    if has_sections:
        structure_score += 0.4 * min(len(sections) / 3, 1.0)
    if has_footer:
        structure_score += 0.2
    score += 0.35 * structure_score
    
    # Content richness (0.35)
    text = soup.get_text(separator=" ", strip=True)
    word_count = len(text.split())
    
    # Ideal word count is 50-150 words per slide
    if word_count < 20:
        content_score = word_count / 20 * 0.3
    elif word_count <= 150:
        content_score = 0.3 + 0.7 * min((word_count - 20) / 80, 1.0)
    else:
        content_score = max(0.5, 1.0 - (word_count - 150) / 200)
    
    # Check for section headings
    headings = soup.select("h2")
    if headings:
        unique_headings = len(set(h.get_text(strip=True) for h in headings))
        content_score += 0.2 * min(unique_headings / 3, 1.0)
    
    score += 0.35 * min(content_score, 1.0)
    
    # Visual elements (0.30)
    visual_score = 0.0
    
    # Check for styling (gradients, shadows, modern CSS)
    style_text = str(soup.select_one("style") or "")
    if "gradient" in style_text:
        visual_score += 0.25
    if "shadow" in style_text:
        visual_score += 0.2
    if "border-radius" in style_text:
        visual_score += 0.15
    if "font-family" in style_text and ("Inter" in style_text or "sans-serif" in style_text):
        visual_score += 0.1
    
    # Check for icons/emojis
    import re
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F9FF"  # Various symbols and pictographs
        "\U00002702-\U000027B0"  # Dingbats
        "\U0001F680-\U0001F6FF"  # Transport and map symbols
        "]+", 
        flags=re.UNICODE
    )
    if emoji_pattern.search(text):
        visual_score += 0.3
    
    score += 0.30 * min(visual_score, 1.0)
    
    return max(0.0, min(1.0, score))


def dino_aesthetic_reward(completions: list, **kwargs) -> list[float]:
    """Score visual aesthetics of all rendered slides.

    Expects kwargs: states (list of SlideForgeState).
    
    Uses DINOv2 for PNG images when available, falls back to HTML heuristics.
    """
    states = kwargs.get("states", [])
    scores = []

    for i, completion in enumerate(completions):
        state = states[i] if i < len(states) else None
        if state is None:
            scores.append(0.0)
            continue
        
        slide_scores = []
        
        # Try PNG-based scoring first
        has_pngs = state.slides_png and any(
            png and png != b"" and str(png) != "b''" 
            for png in state.slides_png
        )
        
        if has_pngs:
            for png_bytes in state.slides_png:
                if not png_bytes or png_bytes == b"" or str(png_bytes) == "b''":
                    slide_scores.append(0.0)
                    continue
                try:
                    s = score_slide_image(png_bytes)
                    slide_scores.append(s)
                except Exception:
                    slide_scores.append(0.0)
        elif state.slides_html:
            # Fallback to HTML-based scoring
            for html in state.slides_html:
                if not html:
                    slide_scores.append(0.0)
                    continue
                try:
                    s = score_slide_html(html)
                    slide_scores.append(s)
                except Exception:
                    slide_scores.append(0.0)

        if slide_scores:
            scores.append(sum(slide_scores) / len(slide_scores))
        else:
            scores.append(0.0)

    return scores
