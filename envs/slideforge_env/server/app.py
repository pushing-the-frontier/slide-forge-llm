"""FastAPI server for SlideForge environment."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..models import SlideForgeAction
from .environment import SlideForgeEnvironment

app = FastAPI(title="SlideForge Environment", version="0.1.0")

_env: Optional[SlideForgeEnvironment] = None


def _get_env() -> SlideForgeEnvironment:
    global _env
    if _env is None:
        _env = SlideForgeEnvironment()
    return _env


# --- Request / Response Models ---

class ResetRequest(BaseModel):
    topic: str = "AI Overview"
    audience: str = "general"
    num_slides: int = 10
    word_count_per_slide: int = 50
    confidence: float = 0.5
    colors: float = 0.5
    sections_per_slide: int = 3
    slide_sections: Optional[dict] = None


class StepRequest(BaseModel):
    tool: str = Field(..., description="Tool name to execute")
    parameters: dict = Field(default_factory=dict, description="Tool parameters")


class ObservationResponse(BaseModel):
    result: str
    success: bool
    current_slide_count: int
    phase: str
    slide_previews: list[str]
    done: bool
    step_number: int
    scores: dict


class StateResponse(BaseModel):
    episode_id: str
    phase: str
    step_count: int
    max_steps: int
    done: bool
    theme: str
    slide_count: int
    outline_count: int
    research_count: int
    accumulated_reward: float


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=ObservationResponse)
def reset(req: ResetRequest):
    env = _get_env()
    brief = req.model_dump()
    obs = env.reset(brief=brief)
    return _obs_to_response(obs)


@app.post("/step", response_model=ObservationResponse)
def step(req: StepRequest):
    env = _get_env()
    action = SlideForgeAction(tool=req.tool, parameters=req.parameters)
    obs = env.step(action)
    return _obs_to_response(obs)


@app.get("/state", response_model=StateResponse)
def get_state():
    env = _get_env()
    s = env.state
    return StateResponse(
        episode_id=s.episode_id,
        phase=s.phase,
        step_count=s.step_count,
        max_steps=s.max_steps,
        done=s.done,
        theme=s.theme,
        slide_count=sum(1 for h in s.slides_html if h),
        outline_count=len(s.outline),
        research_count=len(s.research_context),
        accumulated_reward=s.accumulated_reward,
    )


def _obs_to_response(obs) -> ObservationResponse:
    return ObservationResponse(
        result=obs.result,
        success=obs.success,
        current_slide_count=obs.current_slide_count,
        phase=obs.phase,
        slide_previews=obs.slide_previews,
        done=obs.done,
        step_number=obs.step_number,
        scores=obs.scores,
    )
