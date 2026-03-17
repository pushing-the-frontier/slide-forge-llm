"""SlideForge Environment: reset/step/state following OpenEnv pattern."""

from __future__ import annotations

import os
import uuid
from dataclasses import fields

from ..models import SlideBrief, SlideForgeAction, SlideForgeObservation, SlideForgeState
from .tools import TOOL_REGISTRY

# Valid fields for SlideBrief
_BRIEF_FIELDS = {f.name for f in fields(SlideBrief)}

# LLMs often produce slightly different parameter names; normalise here.
_PARAM_ALIASES: dict[str, str] = {
    "slide_index": "slide_idx",
    "index": "slide_idx",
    "from_index": "from_idx",
    "to_index": "to_idx",
    "theme": "theme_name",
}

# Parameters that must be int (LLMs sometimes emit them as strings)
_INT_PARAMS = {"slide_idx", "from_idx", "to_idx", "slide_index"}


class SlideForgeEnvironment:
    def __init__(self):
        self._state = SlideForgeState()
        self._max_steps = int(os.environ.get("SLIDEFORGE_MAX_STEPS", "50"))

    def reset(
        self,
        brief: dict | None = None,
        existing_slides: list[str] | None = None,
    ) -> SlideForgeObservation:
        # Filter brief to only include valid SlideBrief fields
        if brief:
            filtered_brief = {k: v for k, v in brief.items() if k in _BRIEF_FIELDS}
        else:
            filtered_brief = {}
        
        edit_mode = bool(existing_slides)
        self._state = SlideForgeState(
            episode_id=str(uuid.uuid4()),
            brief=SlideBrief(**filtered_brief) if filtered_brief else SlideBrief(),
            max_steps=self._max_steps,
            edit_mode=edit_mode,
        )

        if existing_slides:
            from .rendering.renderer import render_slide
            self._state.slides_html = list(existing_slides)
            self._state.original_slides_html = list(existing_slides)
            self._state.slides_png = []
            for html in existing_slides:
                png = render_slide(html) if html else b""
                self._state.slides_png.append(png or b"")
            self._state.phase = "REFINE"
            msg = (
                f"Edit mode: loaded {sum(1 for h in existing_slides if h)} existing slides. "
                "Review and improve the presentation."
            )
        else:
            msg = "Episode started. Begin with research or outline creation."

        return self._make_observation(msg, True)

    def step(self, action: SlideForgeAction) -> SlideForgeObservation:
        if self._state.done:
            return self._make_observation("Episode already finished.", False)

        self._state.step_count += 1

        if self._state.step_count > self._state.max_steps:
            self._state.done = True
            return self._make_observation("Max steps reached. Episode terminated.", False)

        result, success = self._execute_tool(action.tool, action.parameters)
        return self._make_observation(result, success)

    @property
    def state(self) -> SlideForgeState:
        return self._state

    def _execute_tool(self, tool_name: str, parameters: dict) -> tuple[str, bool]:
        tool_fn = TOOL_REGISTRY.get(tool_name)
        if tool_fn is None:
            available = ", ".join(TOOL_REGISTRY.keys())
            return f"Unknown tool '{tool_name}'. Available: {available}", False

        normalized: dict = {}
        for k, v in parameters.items():
            canon = _PARAM_ALIASES.get(k, k)
            if canon in _INT_PARAMS and isinstance(v, str) and v.lstrip("-").isdigit():
                v = int(v)
            normalized[canon] = v

        try:
            return tool_fn(self._state, **normalized)
        except TypeError as e:
            return f"Invalid parameters for '{tool_name}': {e}", False
        except Exception as e:
            return f"Tool '{tool_name}' failed: {e}", False

    def _make_observation(self, result: str, success: bool) -> SlideForgeObservation:
        from .rendering.renderer import png_to_base64

        previews = []
        for png in self._state.slides_png:
            if png:
                previews.append(png_to_base64(png))

        return SlideForgeObservation(
            result=result,
            success=success,
            current_slide_count=sum(1 for h in self._state.slides_html if h),
            phase=self._state.phase,
            slide_previews=previews,
            done=self._state.done,
            step_number=self._state.step_count,
        )
