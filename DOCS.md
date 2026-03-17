# SlideForge Technical Documentation

## Table of Contents

1. [Data Models](#data-models)
2. [Environment](#environment)
3. [FastAPI Server](#fastapi-server)
4. [Tools Reference](#tools-reference)
5. [Rendering Pipeline](#rendering-pipeline)
6. [Themes System](#themes-system)
7. [Reward Functions](#reward-functions)
8. [Trajectory Rollouts](#trajectory-rollouts)
9. [Training Pipeline](#training-pipeline)
10. [Client Library](#client-library)
11. [Docker Deployment](#docker-deployment)

---

## Data Models

**File:** `envs/slideforge_env/models.py`

### SlideBrief

Configuration for a presentation generation task. Passed to `reset()`.

| Field | Type | Default | Description |
|---|---|---|---|
| `topic` | `str` | `"AI Overview"` | Presentation topic |
| `audience` | `str` | `"general"` | Target audience |
| `num_slides` | `int` | `10` | Target slide count |
| `word_count_per_slide` | `int` | `50` | Target words per slide |
| `confidence` | `float` | `0.5` | 0.0-1.0. At 1.0, `web_search` is skipped |
| `colors` | `float` | `0.5` | 0.0=grayscale, 1.0=vivid colors |
| `sections_per_slide` | `int` | `3` | Default sections per slide |
| `slide_sections` | `dict\|None` | `None` | Per-slide section overrides, e.g. `{"0": 2, "3": 4}` |

### SlideForgeAction

An agent action = a tool call.

| Field | Type | Description |
|---|---|---|
| `tool` | `str` | Tool name from TOOL_REGISTRY |
| `parameters` | `dict` | Keyword arguments for the tool function |

### SlideForgeObservation

Returned by `reset()` and `step()`.

| Field | Type | Description |
|---|---|---|
| `result` | `str` | Human-readable result of the tool execution |
| `success` | `bool` | Whether the tool call succeeded |
| `current_slide_count` | `int` | Number of non-empty slides |
| `phase` | `str` | Current phase: RESEARCH, PLAN, GENERATE, REFINE, DONE |
| `slide_previews` | `list[str]` | Base64-encoded PNG thumbnails of rendered slides |
| `scores` | `dict` | Live reward component scores (currently empty, populated during training) |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Step reward |
| `reward_details` | `dict` | Per-component reward breakdown |
| `step_number` | `int` | Current step count |

### SlideForgeState

Internal state maintained by the environment across steps.

| Field | Type | Description |
|---|---|---|
| `episode_id` | `str` | UUID for the episode |
| `brief` | `SlideBrief\|None` | Episode configuration |
| `research_context` | `list[dict]` | Accumulated research results |
| `outline` | `list[dict]` | Slide outline entries |
| `slides_html` | `list[str]` | Generated HTML for each slide |
| `slides_png` | `list[bytes]` | Rendered PNG bytes for each slide |
| `theme` | `str` | Current theme name |
| `phase` | `str` | RESEARCH / PLAN / GENERATE / REFINE / DONE |
| `step_count` | `int` | Steps taken this episode |
| `max_steps` | `int` | Max steps before auto-termination (env var: `SLIDEFORGE_MAX_STEPS`) |
| `done` | `bool` | Episode termination flag |
| `accumulated_reward` | `float` | Running reward total |

---

## Environment

**File:** `envs/slideforge_env/server/environment.py`

### SlideForgeEnvironment

```python
class SlideForgeEnvironment:
    def __init__(self)
    def reset(self, brief: dict | None = None) -> SlideForgeObservation
    def step(self, action: SlideForgeAction) -> SlideForgeObservation
    @property
    def state(self) -> SlideForgeState
```

**Lifecycle:**

1. `__init__()` - Creates empty state, reads `SLIDEFORGE_MAX_STEPS` from env (default 50)
2. `reset(brief)` - Creates new `SlideForgeState` with a UUID, sets `SlideBrief` from dict
3. `step(action)` - Increments step counter, executes tool via `TOOL_REGISTRY`, returns observation
4. `state` - Property returning the current `SlideForgeState`

**Termination conditions:**
- Agent calls `finalize` tool (sets `state.done = True`)
- `step_count > max_steps` (auto-terminates)
- Calling `step()` after `done=True` returns "Episode already finished."

**Tool execution:** Each tool function has signature `fn(state: SlideForgeState, **params) -> tuple[str, bool]`. The environment catches `TypeError` (bad params) and general `Exception` gracefully.

---

## FastAPI Server

**File:** `envs/slideforge_env/server/app.py`

### Endpoints

#### `GET /health`
Returns `{"status": "ok"}`. No parameters.

#### `POST /reset`

Start a new episode. Request body (`ResetRequest`):

```json
{
  "topic": "Machine Learning",
  "audience": "engineers",
  "num_slides": 5,
  "word_count_per_slide": 50,
  "confidence": 0.5,
  "colors": 0.7,
  "sections_per_slide": 3,
  "slide_sections": null
}
```

All fields are optional with defaults matching `SlideBrief`. Returns `ObservationResponse`.

#### `POST /step`

Execute a tool. Request body (`StepRequest`):

```json
{
  "tool": "generate_slide",
  "parameters": {
    "slide_idx": 0,
    "title": "Introduction",
    "sections": [{"heading": "Overview", "body": "Content here."}]
  }
}
```

Returns `ObservationResponse`:

```json
{
  "result": "Slide 0 generated (render unavailable, 1 sections).",
  "success": true,
  "current_slide_count": 1,
  "phase": "GENERATE",
  "slide_previews": [],
  "done": false,
  "step_number": 1,
  "scores": {}
}
```

#### `GET /state`

Returns `StateResponse`:

```json
{
  "episode_id": "abc-123",
  "phase": "GENERATE",
  "step_count": 5,
  "max_steps": 50,
  "done": false,
  "theme": "default",
  "slide_count": 2,
  "outline_count": 5,
  "research_count": 1,
  "accumulated_reward": 0.0
}
```

**Note:** The server uses a global singleton `SlideForgeEnvironment`. Only one episode runs at a time.

---

## Tools Reference

**File:** `envs/slideforge_env/server/tools/`

All tools share the same signature: `fn(state: SlideForgeState, **kwargs) -> tuple[str, bool]`

### Research Tools (`tools/research.py`)

#### `web_search(state, query: str)`
- If `state.brief.confidence >= 1.0`, returns "Skipped" immediately
- If `BRAVE_API_KEY` env var is set, calls Brave Search API with 5 results
- If no API key, returns mock results (3 fake entries)
- Appends `{"query": ..., "results": ...}` to `state.research_context`

#### `fetch_url(state, url: str)`
- Fetches URL with httpx (10s timeout, follows redirects)
- Strips script/style/nav/footer/header tags via BeautifulSoup
- Truncates text to 2000 chars
- Appends `{"url": ..., "content": ...}` to `state.research_context`

### Content Tools (`tools/content.py`)

#### `create_outline(state, sections: list[dict])`
- `sections` format: `[{"title": "Slide Title", "bullet_points": ["point1", "point2"]}]`
- Bullet points per slide are capped at `brief.sections_per_slide` (or per-slide override from `brief.slide_sections`)
- Replaces `state.outline` entirely
- Sets `state.phase = "PLAN"`

#### `revise_outline(state, slide_index: int, title: str|None, bullet_points: list[str]|None)`
- Edits a single outline entry in-place
- Returns error if `slide_index` is out of range
- Only updates fields that are provided (partial update)

### Design Tools (`tools/design.py`)

#### `generate_slide(state, slide_idx: int, title: str, sections: list[dict])`
- `sections` format: `[{"heading": "Section Title", "body": "Section content text."}]`
- Generates self-contained HTML (1280x720 viewport) using current theme + color intensity
- Attempts Playwright screenshot; stores PNG bytes if successful
- Auto-expands `state.slides_html` / `state.slides_png` lists if `slide_idx` is beyond current length
- Sets `state.phase = "GENERATE"` (from RESEARCH or PLAN)

#### `edit_slide(state, slide_idx: int, title: str|None, sections: list[dict]|None)`
- Returns error if slide doesn't exist yet
- If `title` or `sections` is `None`, parses existing HTML to preserve current values
- Re-generates HTML and re-renders
- Sets `state.phase = "REFINE"`

#### `set_theme(state, theme_name: str)`
- Valid themes: `default`, `dark`, `corporate`, `creative`, `tech`
- Only changes `state.theme`; does NOT re-render existing slides (call `edit_slide` to re-render)

### Meta Tools (`tools/meta.py`)

#### `review_deck(state)`
- Reports: total slides created vs target, render failures, word count deviations (>30% off target)
- Always returns `success=True` (informational tool)

#### `finalize(state)`
- Returns error if no slides exist
- Sets `state.phase = "DONE"` and `state.done = True`
- Episode ends after this call

---

## Rendering Pipeline

**Files:** `envs/slideforge_env/server/rendering/`

### HTML Generator (`html_generator.py`)

`generate_slide_html(title, sections, theme_name, color_intensity, slide_index, total_slides) -> str`

- Produces a self-contained HTML document targeting 1280x720 pixels
- Uses inline CSS (no external dependencies)
- Font: Segoe UI / Arial fallback
- Structure: `.title` div, `.sections` flex container with `.section` children, `.footer` with slide number
- Colors resolved from theme + intensity at generation time

### Renderer (`renderer.py`)

- `render_slide(html) -> bytes|None` - Synchronous wrapper
- `render_slide_async(html) -> bytes|None` - Async Playwright rendering
- Uses a global browser instance (lazy-initialized, reused across calls)
- Launches headless Chromium with `--no-sandbox --disable-gpu`
- Timeout controlled by `SLIDEFORGE_RENDER_TIMEOUT` env var (default 30s)
- Returns `None` on any failure (Playwright not installed, render error, timeout)
- `png_to_base64(png_bytes) -> str` - Base64 encoding helper

**Graceful degradation:** If Playwright is not installed, `render_slide()` returns `None` and the environment continues without screenshots. Slides are still stored as HTML.

---

## Themes System

**File:** `envs/slideforge_env/server/rendering/themes.py`

### Color Intensity Interpolation

The `colors` parameter in `SlideBrief` controls color saturation:

- `colors=0.0` - All theme colors converted to grayscale (luminance-weighted: 0.299R + 0.587G + 0.114B)
- `colors=0.5` - Midpoint between grayscale and the theme's defined colors
- `colors=1.0` - Full vivid theme colors as defined

`resolve_theme(theme_name: str, color_intensity: float) -> dict[str, str]` returns CSS-ready `rgb()` strings for keys: `name`, `bg`, `text`, `accent`, `secondary`.

### Theme Definitions

| Theme | Background | Text | Accent | Secondary |
|---|---|---|---|---|
| `default` | White (255,255,255) | Dark gray (33,33,33) | Blue (41,98,255) | Light blue (100,181,246) |
| `dark` | Near-black (30,30,30) | Near-white (240,240,240) | Green (0,200,83) | Medium green (76,175,80) |
| `corporate` | Light gray (245,245,245) | Navy (44,62,80) | Slate (52,73,94) | Silver (149,165,166) |
| `creative` | Warm white (255,253,231) | Dark gray (33,33,33) | Orange (255,87,34) | Amber (255,167,38) |
| `tech` | Near-black (18,18,18) | Light gray (224,224,224) | Cyan (0,229,255) | Teal (29,233,182) |

---

## Reward Functions

**Files:** `rewards/`

All reward functions follow the GRPO-compatible signature:

```python
def reward_fn(completions: list, **kwargs) -> list[float]
```

They expect `kwargs["states"]` to be a list of `SlideForgeState` objects (one per completion).

### code_rules (`rewards/code_rules.py`)

Per-slide scoring (averaged across all slides), each worth 0.25:

| Check | Score | Criteria |
|---|---|---|
| Title present | 0.25 | `.title` element exists and has text |
| Section count | 0.25 (exact) / 0.1 (any) | Matches `brief.sections_per_slide` (or per-slide override) |
| Word count | 0.25 * ratio | `min(actual, target) / max(actual, target)` |
| Non-empty sections | 0.25 * (filled/total) | Sections that contain text |

**Range:** [0.0, 1.0]

### render_quality (`rewards/render_quality.py`)

| Component | Weight | Criteria |
|---|---|---|
| Coverage | 0.4 | `min(created_slides / target_slides, 1.0)` |
| Render success | 0.3 | `rendered_count / created_count` |
| HTML validity | 0.3 | Has `.title` and `.section` elements |

**Range:** [0.0, 1.0]

### dino_aesthetic (`rewards/dino_aesthetic.py`)

Uses `facebook/dinov2-small` (lazy-loaded, GPU if available). Scores each rendered PNG:

| Metric | Weight | What it measures |
|---|---|---|
| Feature diversity | 0.4 | Std dev of patch token features (layout variety) |
| Spatial coherence | 0.35 | Cosine similarity between adjacent patch tokens (visual smoothness) |
| Activation magnitude | 0.25 | Norm of patch features (visual richness) |

- Model: DINOv2-small (facebook/dinov2-small) via HuggingFace transformers
- Input: PNG screenshot -> PIL Image -> DINOv2 processor
- Processes patch tokens (skips CLS token)
- Falls back to 0.0 on any error (torch not installed, CUDA OOM, etc.)

**Range:** [0.0, 1.0]

**Requirements:** `torch`, `transformers`, `Pillow`

### content_quality (`rewards/content_quality.py`)

| Component | Weight | Criteria |
|---|---|---|
| Topic relevance | 0.35 | Fraction of slides mentioning topic words |
| Factual grounding | 0.25 | Overlap between slide text and research results (>3 non-stopword matches) |
| Content uniqueness | 0.20 | Ratio of unique slide texts |
| Narrative flow | 0.20 | Outline coverage relative to target |

- If no research was done and `confidence < 0.8`: grounding score = 0.0
- If no research was done and `confidence >= 0.8`: grounding score = 0.15 (assumed high-quality)

**Range:** [0.0, 1.0]

### Aggregator (`rewards/aggregator.py`)

```python
DEFAULT_WEIGHTS = {
    "code_rules": 1.0,
    "render_quality": 2.0,
    "dino_aesthetic": 3.0,
    "content_quality": 2.0,
}
```

- `aggregate_rewards(completions, weights=None, **kwargs) -> list[float]` - Single weighted score
- `compute_reward_details(completions, weights=None, **kwargs) -> list[dict]` - Per-component breakdown with `"aggregate"` key

Formula: `sum(weight_i * score_i) / sum(all_weights)`

To exclude a component (e.g., skip DINOv2 when torch isn't available), pass a custom weights dict without that key:

```python
aggregate_rewards(completions, weights={"code_rules": 1.0, "render_quality": 2.0, "content_quality": 2.0}, states=[state])
```

Any component that raises an exception is scored as 0.0 for all completions.

---

## Trajectory Rollouts

**File:** `training/rollouts.py`

Generates multi-step expert trajectories by running Claude Opus 4.6 (via AWS Bedrock Converse API) as the SlideForge agent. Each rollout is a full episode: research -> outline -> generate slides -> review -> finalize.

### Architecture

```
Claude Opus 4.6 (Bedrock)
        |
   assistant text (with JSON tool call)
        |
        v
  extract_tool_call()
        |
        v
  SlideForgeEnvironment.step()
        |
   observation text
        |
        v
  append to conversation, loop
        |
        v
  compute rewards, store trajectory
```

### API: `run_rollout()`

```python
def run_rollout(
    brief: dict,                          # SlideBrief params
    client=None,                          # boto3 bedrock-runtime client (auto-created if None)
    model_id: str = "us.anthropic.claude-opus-4-6-20250514-v1:0",
    max_turns: int = 30,                  # Max agent turns per episode
    temperature: float = 0.7,
    reward_weights: dict | None = None,   # Custom reward weights (excludes dino by default)
    verbose: bool = False,
) -> dict
```

Returns a trajectory dict with keys:

| Key | Type | Description |
|---|---|---|
| `episode_id` | `str` | UUID for the episode |
| `brief` | `dict` | The input brief |
| `turns` | `list[dict]` | Per-turn records (see below) |
| `total_steps` | `int` | Environment steps taken |
| `final_phase` | `str` | Last phase reached |
| `completed` | `bool` | True if agent called `finalize` successfully |
| `slides_created` | `int` | Number of non-empty slides |
| `theme` | `str` | Final theme |
| `rewards` | `dict` | Per-component reward scores + aggregate |
| `messages` | `list[dict]` | Full conversation in `[{role, content}]` format |
| `elapsed_seconds` | `float` | Wall-clock time for the episode |

Each turn record:

| Key | Type | Description |
|---|---|---|
| `turn` | `int` | Turn index |
| `assistant` | `str` | Raw Claude output text |
| `tool_call` | `dict\|None` | Extracted `{tool, parameters}` or None |
| `observation` | `str` | Environment result text |
| `success` | `bool` | Whether the tool call succeeded |
| `phase` | `str` | Environment phase after the step |
| `slide_count` | `int` | Slides created so far |
| `done` | `bool` | Whether the episode ended |

### API: `run_batch()`

```python
def run_batch(
    briefs: list[dict] | None = None,     # Defaults to 10 built-in briefs
    num_rollouts_per_brief: int = 1,
    model_id: str = ...,
    max_turns: int = 30,
    temperature: float = 0.7,
    reward_weights: dict | None = None,
    verbose: bool = True,
) -> list[dict]
```

Runs `len(briefs) * num_rollouts_per_brief` episodes sequentially, reusing a single Bedrock client.

### API: `trajectories_to_dataset()`

```python
def trajectories_to_dataset(trajectories: list[dict]) -> datasets.Dataset
```

Converts trajectories to a flat HuggingFace Dataset with columns:

`episode_id`, `brief`, `topic`, `audience`, `num_slides_target`, `slides_created`, `completed`, `total_steps`, `final_phase`, `theme`, `num_turns`, `reward_code_rules`, `reward_render_quality`, `reward_content_quality`, `reward_aggregate`, `messages` (JSON string), `turns` (JSON string), `elapsed_seconds`

### API: `push_to_hub()`

```python
def push_to_hub(
    trajectories: list[dict],
    repo_id: str,                          # e.g. "username/slideforge-trajectories"
    private: bool = True,
    token: str | None = None,              # Falls back to HF_TOKEN env var
) -> datasets.Dataset
```

### Bedrock Configuration

- **Model ID:** `us.anthropic.claude-opus-4-6-20250514-v1:0`
- **API:** Bedrock Converse (`client.converse()`)
- **Region:** `AWS_REGION` env var or `us-east-1`
- **Auth:** Standard boto3 credential chain (env vars, `~/.aws/credentials`, IAM role, etc.)
- **Max tokens per turn:** 1024
- **System prompt:** High-level strategy prompt guiding Claude through the research -> outline -> generate -> review -> finalize workflow

### System Prompt

The rollout system prompt instructs Claude to:
1. Research the topic (if confidence < 1.0) using `web_search`
2. Create an outline covering all target slides
3. Optionally set a theme
4. Generate each slide with proper sections and word count
5. Review with `review_deck` and fix issues with `edit_slide`
6. Call `finalize` when done
7. Output exactly ONE JSON tool call per turn in `` ```json ... ``` `` markers

### Recovery

If Claude fails to produce a valid tool call on a turn, the system sends a retry message: "You must respond with a JSON tool call. Please try again." This gives Claude one chance to recover before the turn is recorded as a failure.

### CLI

```
usage: rollouts.py [-h] [--briefs-file BRIEFS_FILE] [--num-rollouts N]
                   [--max-turns N] [--temperature T] [--model-id ID]
                   [--region REGION] [--output PATH]
                   [--push-to-hub REPO_ID] [--hf-token TOKEN]
                   [--private] [--public] [--quiet]
```

| Argument | Default | Description |
|---|---|---|
| `--briefs-file` | None | JSON file with briefs array |
| `--num-rollouts` | `1` | Rollouts per brief |
| `--max-turns` | `30` | Max agent turns per episode |
| `--temperature` | `0.7` | Claude sampling temperature |
| `--model-id` | `us.anthropic.claude-opus-4-6-20250514-v1:0` | Bedrock model ID |
| `--region` | `us-east-1` | AWS region |
| `--output` | `outputs/trajectories.json` | Local JSON output path |
| `--push-to-hub` | None | HuggingFace repo ID |
| `--hf-token` | None | HF token (or `HF_TOKEN` env) |
| `--private` | True | Make HF dataset private |
| `--public` | False | Make HF dataset public |
| `--quiet` | False | Suppress verbose output |

### Using Trajectories for Training

The `messages` column is in standard chat format and can be used directly for SFT:

```python
from datasets import load_dataset
import json

ds = load_dataset("username/slideforge-trajectories")

# Filter to completed, high-reward trajectories
good = ds.filter(lambda x: x["completed"] and x["reward_aggregate"] > 0.7)

# Extract messages for SFT
for row in good:
    messages = json.loads(row["messages"])
    # messages = [{"role": "system", "content": ...}, {"role": "user", ...}, ...]
```

---

## Training Pipeline

**Files:** `training/`

### Prompt Template (`prompts.py`)

`AGENT_PROMPT` is a format string with placeholders: `{topic}`, `{audience}`, `{num_slides}`, `{word_count_per_slide}`, `{sections_per_slide}`, `{colors}`, `{confidence}`, `{phase}`, `{current_slide_count}`, `{research_count}`.

`format_prompt(state: SlideForgeState) -> str` fills the template from state.

### Tool Call Extraction (`grpo_trainer.py`)

`extract_tool_call(text: str) -> dict | None`

Parsing priority:
1. `` ```json {...} ``` `` fenced code block
2. Raw JSON with `"tool"` key, balanced brace matching for nested objects
3. Returns `None` if no valid tool call found

### Reward Function (`grpo_trainer.py`)

`slideforge_reward(completions, **kwargs) -> list[float]`

For each completion:
1. Extract tool call from text
2. If no tool call found: score = -2.0
3. Create fresh `SlideForgeEnvironment`, reset with brief, execute one step
4. If step fails: score = -1.0
5. If step succeeds: score = `aggregate_rewards()` result

### Trainer Setup (`grpo_trainer.py`)

`create_trainer(model_name, max_seq_length, lora_r, learning_rate, num_generations, max_steps)`

Returns `(model, tokenizer, training_args)`. Uses:
- `unsloth.FastLanguageModel` for 4-bit model loading
- LoRA on: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- `trl.GRPOConfig` with specified hyperparameters
- Output dir: `outputs/slideforge_grpo`

### Training Script (`run_training.py`)

CLI arguments:

| Argument | Default | Description |
|---|---|---|
| `--model` | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` | HuggingFace model ID |
| `--max-seq-length` | `2048` | Max sequence length |
| `--lora-r` | `16` | LoRA rank |
| `--lr` | `5e-5` | Learning rate |
| `--num-generations` | `2` | GRPO generations per prompt |
| `--max-steps` | `1000` | Training steps |
| `--briefs-file` | `None` | Path to JSON file with custom briefs |

Built-in dataset: 8 example briefs spanning AI, climate, ML, startup, privacy, quantum, remote work, cybersecurity topics.

---

## Client Library

**File:** `envs/slideforge_env/client.py`

```python
from envs.slideforge_env.client import SlideForgeClient
from envs.slideforge_env.models import SlideForgeAction

with SlideForgeClient("http://localhost:8000") as client:
    client.health()                          # -> {"status": "ok"}
    client.reset(topic="AI", num_slides=5)   # -> ObservationResponse dict
    client.step(SlideForgeAction(            # -> ObservationResponse dict
        tool="web_search",
        parameters={"query": "AI trends"}
    ))
    client.state()                           # -> StateResponse dict
```

Uses httpx with 60-second timeout. Supports context manager for automatic cleanup.

---

## Docker Deployment

**File:** `envs/slideforge_env/server/Dockerfile`

```
Base:     python:3.11-slim
Installs: curl, pip deps from requirements.txt, playwright + chromium
Copies:   envs/ and rewards/ directories
Env vars: SLIDEFORGE_MAX_STEPS=50, SLIDEFORGE_RENDER_TIMEOUT=30
Health:   curl -f http://localhost:8000/health (every 30s)
CMD:      uvicorn on 0.0.0.0:8000
```

Build context must be the repo root (so COPY paths resolve correctly):

```bash
docker build -f envs/slideforge_env/server/Dockerfile -t slideforge:latest .
docker run -p 8000:8000 -e BRAVE_API_KEY=your_key slideforge:latest
```
