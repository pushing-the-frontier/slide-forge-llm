from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SlideBrief:
    topic: str = "AI Overview"
    audience: str = "general"
    num_slides: int = 10
    word_count_per_slide: int = 50
    confidence: float = 0.5
    colors: float = 0.5
    sections_per_slide: int = 3
    slide_sections: Optional[dict] = None
    edit_instructions: str = ""


@dataclass
class SlideForgeAction:
    tool: str
    parameters: dict = field(default_factory=dict)


@dataclass
class SlideForgeObservation:
    result: str
    success: bool
    current_slide_count: int = 0
    phase: str = "RESEARCH"
    slide_previews: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    done: bool = False
    reward: float = 0.0
    reward_details: dict = field(default_factory=dict)
    step_number: int = 0


@dataclass
class SlideForgeState:
    episode_id: str = ""
    brief: Optional[SlideBrief] = None
    research_context: list = field(default_factory=list)
    outline: list = field(default_factory=list)
    slides_html: list = field(default_factory=list)
    slides_png: list = field(default_factory=list)
    theme: str = "default"
    phase: str = "RESEARCH"
    step_count: int = 0
    max_steps: int = 50
    done: bool = False
    accumulated_reward: float = 0.0
    edit_mode: bool = False
    original_slides_html: list = field(default_factory=list)
