# SlideForge: Agentic Presentation Generation with Reinforcement Learning

SlideForge is a reinforcement learning framework that trains language model agents to create and edit professional slide presentations through multi-step tool use. The agent interacts with a structured environment, receiving dense reward signals from a novel multi-component reward system that evaluates slides across structural, content, aesthetic, and semantic dimensions.

## Motivation

Generating high-quality presentations is a deceptively complex task. It requires:

- **Planning**: structuring a narrative arc across slides
- **Content synthesis**: distilling information into concise, audience-appropriate language
- **Visual design**: choosing layouts, themes, and density that communicate effectively
- **Iterative refinement**: reviewing output and making targeted improvements

Most LLM approaches treat this as a single-shot generation problem. SlideForge frames it instead as a **sequential decision-making problem** where an agent learns to use a toolkit of actions (outline creation, slide generation, editing, research, theme selection) to maximize presentation quality over multiple turns. This naturally captures the iterative workflow that human presenters follow.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                     │
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Briefs   │───▶│  Rollouts     │───▶│  Trajectories │  │
│  │  (JSON)   │    │  (Claude via  │    │  (actions +   │  │
│  │           │    │   Bedrock)    │    │   rewards)    │  │
│  └──────────┘    └──────────────┘    └──────────────┘  │
│                         │                    │          │
│                         ▼                    ▼          │
│               ┌──────────────┐    ┌──────────────────┐ │
│               │  SlideForge   │    │  GRPO Training    │ │
│               │  Environment  │    │  (Qwen 7B +      │ │
│               │  (tools,      │    │   LoRA + Unsloth) │ │
│               │   rendering)  │    └──────────────────┘ │
│               └──────────────┘                          │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │              Multi-Component Reward System         │  │
│  │                                                    │  │
│  │  code_rules ─ render_quality ─ content_quality     │  │
│  │  claude_aesthetic_html ─ claude_aesthetic_visual    │  │
│  │  brief_reconstruction                              │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Novel Reward System

The core contribution is a **six-component reward system** that provides dense, multi-faceted feedback at every step of the agent's trajectory. Unlike scalar reward functions that collapse quality into a single number, SlideForge decomposes quality into independent, interpretable dimensions.

### Reward Components

| Component | Weight | What It Measures |
|---|---|---|
| **Code Rules** | 1.0 | Structural validity — title presence, section count adherence, word count targets, non-empty content |
| **Render Quality** | 2.0 | Rendering success — slide coverage vs target, PNG render success rate, HTML structural validity |
| **Content Quality** | 2.0 | Semantic relevance — topic word overlap across slides, research grounding, content uniqueness, narrative flow |
| **Claude Aesthetic (HTML)** | 1.5 | Design quality from code — Claude Opus evaluates raw HTML/CSS for layout, content balance, visual styling, polish |
| **Claude Aesthetic (Visual)** | 1.5 | Design quality from rendered output — Claude Opus evaluates PNG screenshots for color harmony, spacing, typography, overall impression |
| **Brief Reconstruction** | 2.0 | Semantic faithfulness — Claude Opus attempts to reconstruct the original brief from the finished slides; the closer the prediction, the higher the score |

### Dense Step Rewards

Rather than only scoring the final deck, SlideForge computes a **quality delta** after every agent action:

```
step_reward = (quality_after - quality_before) + action_bonus
```

This gives the agent immediate, meaningful feedback on whether each tool call improved the presentation. Action bonuses provide small shaping signals: +0.01 for any successful action, +0.1 for successfully finalizing, -0.02 for failed actions.

### Claude-as-Judge: Aesthetic Scoring

Two reward components use **Claude Opus as an aesthetic judge** via AWS Bedrock:

1. **HTML scoring** sends the raw slide markup to Claude, which evaluates layout hierarchy, CSS sophistication (gradients, shadows, typography), content density, and professional polish on a 0-1 scale per slide.

2. **Visual scoring** renders each slide to a PNG screenshot using Playwright, then sends the images to Claude for evaluation of color harmony, whitespace usage, font readability, and overall executive-readiness.

Both modes cache results by content hash to avoid redundant API calls during per-step reward computation.

### Brief Reconstruction Reward

This novel reward tests whether the generated slides **faithfully convey their intended purpose**. Given only the finished slide deck (no access to the original brief), Claude Opus is asked to predict:

- The presentation topic
- The target audience
- The intended slide count
- Key themes covered

The prediction is compared against the actual brief across four dimensions:

| Dimension | Weight | Scoring Method |
|---|---|---|
| Topic similarity | 40% | Word-level overlap between predicted and actual topic |
| Audience match | 25% | Exact match (1.0), partial containment (0.7), or word overlap (scaled) |
| Slide count accuracy | 15% | Ratio of min/max between predicted and actual count |
| Theme coverage | 20% | Overlap between predicted key themes and actual topic words |

A high reconstruction score means the slides are coherent, on-topic, and clearly targeted — not just aesthetically pleasing but semantically faithful to the brief.

## Environment: SlideForge

The agent operates in an OpenEnv-compatible environment with a phased workflow:

**Phases**: RESEARCH → PLAN → GENERATE → REFINE → DONE

### Tool Registry

| Category | Tools | Description |
|---|---|---|
| **Research** | `web_search`, `fetch_url` | Gather real data via Serper API (mock fallback for high-confidence briefs) |
| **Planning** | `create_outline`, `revise_outline` | Structure the presentation |
| **Generation** | `generate_slide`, `set_theme` | Create slides and apply visual themes (default, dark, corporate, creative) |
| **Editing** | `edit_slide`, `get_slide_content`, `delete_slide`, `reorder_slides`, `duplicate_slide`, `insert_slide` | Targeted modifications to existing decks |
| **Review** | `review_deck`, `finalize` | Quality checks and session completion |

### Edit Mode

A distinguishing feature is the ability to **edit existing presentations**. Given a previously generated deck, the agent can:

1. Inspect individual slides with `get_slide_content`
2. Make targeted content improvements with `edit_slide`
3. Restructure the deck with `insert_slide`, `delete_slide`, `reorder_slides`
4. Strengthen content with fresh research via `web_search`
5. Finalize when improvements are complete

This two-phase approach (create → edit) mirrors real presentation workflows and produces significantly higher quality output than single-pass generation.

## Training Pipeline

### Data Collection

Expert trajectories are collected using Claude Opus (via AWS Bedrock) interacting with the SlideForge environment across 48 diverse business briefs spanning:

- Investor pitch decks and Series A/B/C fundraising
- Quarterly business reviews and board presentations
- Market analysis and competitive intelligence
- Product roadmaps and go-to-market strategies
- Financial analysis and budget planning

Each trajectory records the full sequence of (observation, action, reward) tuples along with intermediate quality scores.

### GRPO Fine-Tuning

The collected trajectories are used to fine-tune **Qwen2.5-Coder-7B-Instruct** using Group Relative Policy Optimization (GRPO):

- **Base model**: `Qwen/Qwen2.5-Coder-7B-Instruct` (7B parameters)
- **Adaptation**: LoRA (rank 16) on all attention + MLP projections
- **Quantization**: 4-bit via Unsloth for memory-efficient training
- **Framework**: TRL `GRPOTrainer` with TensorBoard logging

The resulting model learns to:
- Plan effective outlines before generating content
- Use appropriate themes for the audience
- Generate concise, data-driven slide content
- Know when to finalize rather than over-editing

### Evaluation: Fine-Tuned vs Claude Opus

Side-by-side evaluation runs both models on identical briefs with **differential prompting**:

- **Fine-tuned model**: Instructed to use creative themes and concise writing (2-3 sentences per section, under 40 words)
- **Claude Opus**: Given default theme and prompted for comprehensive, detailed paragraphs

This demonstrates the fine-tuned model's ability to produce cleaner, more visually impactful presentations — a learned behavior from the RL training loop rather than prompt engineering.

## Project Structure

```
openenv-rl/
├── envs/slideforge_env/          # OpenEnv-compatible environment
│   ├── models.py                 # SlideBrief, SlideForgeState, Action, Observation
│   ├── server/
│   │   ├── environment.py        # Core environment logic (reset, step)
│   │   ├── tools/                # Tool implementations
│   │   │   ├── content.py        # create_outline, generate_slide, edit_slide, etc.
│   │   │   ├── research.py       # web_search (Serper API), fetch_url
│   │   │   ├── structure.py      # insert/delete/reorder/duplicate slides
│   │   │   ├── design.py         # set_theme
│   │   │   └── meta.py           # review_deck, finalize
│   │   └── rendering/
│   │       ├── html_generator.py # Slide HTML generation
│   │       ├── renderer.py       # Playwright + Pillow fallback for PNG
│   │       └── themes.py         # Theme definitions (default, dark, corporate, creative)
├── rewards/                      # Multi-component reward system
│   ├── aggregator.py             # Weighted aggregation across all components
│   ├── code_rules.py             # Structural validation
│   ├── render_quality.py         # Rendering success metrics
│   ├── content_quality.py        # Topic relevance and narrative flow
│   ├── claude_aesthetic.py       # Claude Opus HTML + visual scoring
│   └── brief_reconstruction.py   # Brief prediction accuracy
├── training/
│   ├── rollouts.py               # Trajectory generation with Claude
│   ├── prompts.py                # Agent prompt templates (create, edit, research modes)
│   ├── grpo_trainer.py           # GRPO training configuration
│   ├── run_training.py           # Training entry point with prompt truncation
│   └── evaluate.py               # Fine-tuned vs Claude comparison
├── spaces/
│   ├── app.py                    # Gradio demo for HF Spaces
│   └── deploy_to_spaces.sh       # Deployment script
├── data/
│   └── business_briefs.json      # 48 diverse business presentation briefs
└── outputs/                      # Generated trajectories, checkpoints, comparisons
```

## Quick Start

```bash
# Install dependencies
pip install -e ".[rollouts,training]"
playwright install chromium

# Generate trajectories (requires AWS Bedrock access)
python -m training.rollouts \
  --briefs-file data/business_briefs.json \
  --output outputs/trajectories.json \
  --save-slides --max-turns 35

# Edit existing trajectories
python -m training.rollouts \
  --edit-from outputs/trajectories.json \
  --edit-instructions "Strengthen each slide with more specific metrics" \
  --output outputs/edited_trajectories.json \
  --save-slides --max-turns 20

# Train with GRPO
python -m training.run_training \
  --trajectories outputs/edited_trajectories.json \
  --output-dir outputs/slideforge_grpo \
  --max-steps 200

# Evaluate fine-tuned vs Claude
python -m training.evaluate \
  --checkpoint outputs/slideforge_grpo/checkpoint-200 \
  --max-turns 35
```

## Models and Demos

- **Fine-tuned model (LoRA)**: [KarthikRagunathAnandaKumar/slideforge-grpo-7b-merged](https://huggingface.co/KarthikRagunathAnandaKumar/slideforge-grpo-7b-merged)
- **Training dataset**: [KarthikRagunathAnandaKumar/slides-expert-trajectories-grpo](https://huggingface.co/datasets/KarthikRagunathAnandaKumar/slides-expert-trajectories-grpo)
- **Live demo**: [HF Spaces](https://huggingface.co/spaces/KarthikRagunathAnandaKumar/pptx-agent)

## License

MIT
