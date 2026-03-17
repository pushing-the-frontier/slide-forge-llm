"""GRPO trainer adapted from the Unsloth notebook pattern for SlideForge."""

from __future__ import annotations

import json
import re
from typing import Optional

from envs.slideforge_env.models import SlideForgeAction, SlideBrief
from envs.slideforge_env.server.environment import SlideForgeEnvironment
from rewards.aggregator import aggregate_rewards, compute_reward_details


def extract_tool_call(text: str) -> Optional[dict]:
    """Extract a JSON tool call from LLM output."""
    # Try ```json ... ``` block
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON object with nested braces
    match = re.search(r"\{[^{}]*\"tool\"\s*:\s*\"[^\"]+\".*?\}", text, re.DOTALL)
    if match:
        # Find the balanced closing brace
        s = match.start()
        depth = 0
        for j in range(s, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[s:j + 1])
                    except json.JSONDecodeError:
                        break

    return None


def run_episode(env: SlideForgeEnvironment, actions: list[dict], brief: dict) -> tuple[list, float]:
    """Run a full episode with a sequence of tool calls. Returns (observations, total_reward)."""
    env.reset(brief=brief)
    observations = []

    for action_dict in actions:
        tool = action_dict.get("tool", "")
        params = action_dict.get("parameters", {})
        action = SlideForgeAction(tool=tool, parameters=params)
        obs = env.step(action)
        observations.append(obs)
        if obs.done:
            break

    return observations, env.state.accumulated_reward


def slideforge_reward(completions: list, **kwargs) -> list[float]:
    """Reward function compatible with GRPO trainer.

    Each completion is a list of message dicts. We extract tool calls,
    run them through the environment, and compute rewards.
    """
    briefs = kwargs.get("briefs", [{"topic": "AI Overview"}] * len(completions))
    scores = []
    states = []

    for i, completion in enumerate(completions):
        env = SlideForgeEnvironment()
        brief = briefs[i] if i < len(briefs) else {"topic": "AI Overview"}
        env.reset(brief=brief)

        # Extract and execute all tool calls from the completion
        if isinstance(completion, list):
            text = " ".join(m.get("content", "") for m in completion if isinstance(m, dict))
        else:
            text = str(completion)

        tool_call = extract_tool_call(text)
        if tool_call is None:
            scores.append(-2.0)
            states.append(env.state)
            continue

        try:
            action = SlideForgeAction(
                tool=tool_call.get("tool", ""),
                parameters=tool_call.get("parameters", {}),
            )
            obs = env.step(action)
            states.append(env.state)

            if not obs.success:
                scores.append(-1.0)
            else:
                scores.append(1.0)
        except Exception:
            scores.append(-1.0)
            states.append(env.state)

    # Compute aggregate rewards for episodes that have slides
    agg_scores = aggregate_rewards(completions, states=states)
    for i in range(len(scores)):
        if scores[i] > 0:
            scores[i] = agg_scores[i]

    return scores


def create_trainer(
    model_name: str = "unsloth/Qwen2.5-Coder-7B-Instruct",
    max_seq_length: int = 8192,
    lora_r: int = 16,
    learning_rate: float = 5e-5,
    num_generations: int = 2,
    max_steps: int = 1000,
):
    """Create and return a configured GRPO trainer.

    Requires: unsloth, trl, peft (install with [training] extra).
    """
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    training_args = GRPOConfig(
        temperature=1.0,
        learning_rate=learning_rate,
        num_generations=num_generations,
        max_completion_length=1024,
        max_steps=max_steps,
        save_steps=25,
        output_dir="outputs/slideforge_grpo",
        logging_steps=2,
        logging_dir="outputs/slideforge_grpo/logs",
        report_to="tensorboard",
    )

    return model, tokenizer, training_args
