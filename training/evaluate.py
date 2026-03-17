"""Compare fine-tuned model vs Claude on SlideForge presentations.

Runs both models on the same briefs from the training dataset, generates
deck.html + deck.pptx for each, and outputs a side-by-side comparison.

Usage:
    python -m training.evaluate \
        --checkpoint outputs/slideforge_grpo/checkpoint-200 \
        --trajectories outputs/all_48_trajectories.json \
        --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
logging.disable(logging.WARNING)

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import boto3
import torch

logging.disable(logging.NOTSET)

from envs.slideforge_env.models import SlideForgeAction, SlideForgeState, SlideBrief
from envs.slideforge_env.server.environment import SlideForgeEnvironment
from rewards.aggregator import compute_reward_details, DEFAULT_WEIGHTS
from training.grpo_trainer import extract_tool_call
from training.prompts import format_prompt
from training.rollouts import (
    SYSTEM_PROMPT,
    BEDROCK_MODEL_ID,
    BEDROCK_REGION,
    MAX_TOKENS_PER_TURN,
    compute_quality_score,
    compute_step_reward,
    _call_claude,
    _create_bedrock_client,
    _build_deck_html,
    _build_deck_pptx,
)

OUTPUT_BASE = "outputs/output_all48"

BEDROCK_MODELS = {
    "claude-opus": {
        "model_id": "us.anthropic.claude-opus-4-6-v1",
        "label": "Claude Opus 4.6",
    },
    "claude-sonnet": {
        "model_id": "us.anthropic.claude-sonnet-4-6",
        "label": "Claude Sonnet 4.6",
    },
    "gpt-oss-120b": {
        "model_id": "openai.gpt-oss-120b-1:0",
        "label": "GPT OSS 120B",
    },
    "llama4-scout": {
        "model_id": "us.meta.llama4-scout-17b-instruct-v1:0",
        "label": "Llama 4 Scout 17B",
    },
}


def load_finetuned_model(checkpoint_path: str, max_seq_length: int = 8192):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_finetuned(model, tokenizer, messages: list[dict],
                       max_new_tokens: int = 512, max_seq_length: int = 8192) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    max_prompt_tokens = max_seq_length - max_new_tokens
    orig_len = inputs["input_ids"].shape[1]
    if orig_len > max_prompt_tokens:
        inputs["input_ids"] = inputs["input_ids"][:, -max_prompt_tokens:]
        inputs["attention_mask"] = inputs["attention_mask"][:, -max_prompt_tokens:]
    print(f"    [generating... {inputs['input_ids'].shape[1]} prompt tokens]", flush=True)

    _prev_level = logging.root.manager.disable
    logging.disable(logging.ERROR)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    logging.disable(_prev_level)

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _slug(topic: str) -> str:
    s = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)
    return s.replace(" ", "_").lower()[:50]


def save_deck(trajectory: dict, output_dir: str):
    """Save deck.html, deck.pptx, brief.json, and quality.json to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    if trajectory.get("slides_html"):
        with open(f"{output_dir}/deck.html", "w") as f:
            f.write(_build_deck_html(trajectory))
        try:
            _build_deck_pptx(trajectory, f"{output_dir}/deck.pptx")
        except Exception as e:
            print(f"    Warning: PPTX export failed: {e}")

        slides_dir = f"{output_dir}/slides"
        os.makedirs(slides_dir, exist_ok=True)
        for idx, html in enumerate(trajectory.get("slides_html", [])):
            if html:
                with open(f"{slides_dir}/slide_{idx+1}.html", "w") as f:
                    f.write(html)

    with open(f"{output_dir}/brief.json", "w") as f:
        json.dump(trajectory["brief"], f, indent=2)

    with open(f"{output_dir}/quality.json", "w") as f:
        json.dump(trajectory["final_quality"], f, indent=2, default=str)

    if trajectory.get("steps"):
        with open(f"{output_dir}/trajectory.json", "w") as f:
            json.dump({
                "model": trajectory.get("model"),
                "brief_topic": trajectory.get("brief_topic"),
                "completed": trajectory.get("completed"),
                "turns_used": trajectory.get("turns_used"),
                "cumulative_reward": trajectory.get("cumulative_reward"),
                "elapsed_seconds": trajectory.get("elapsed_seconds", 0),
                "final_quality": trajectory.get("final_quality"),
                "steps": trajectory["steps"],
            }, f, indent=2, default=str)


def run_episode(model_name: str, brief: dict, max_turns: int = 30,
                model=None, tokenizer=None, bedrock_client=None,
                bedrock_model_id: str | None = None,
                verbose: bool = True) -> dict:
    """Run one full episode through the SlideForge environment.

    For Bedrock models, pass bedrock_client + bedrock_model_id.
    For local HF models, pass model + tokenizer.
    """
    is_bedrock = bedrock_model_id is not None
    env = SlideForgeEnvironment()
    obs = env.reset(brief=brief)

    content = brief.get("content") if isinstance(brief, dict) else None
    user_context = format_prompt(env.state, content=content)
    theme = brief.get("preferred_theme")

    context = user_context
    if theme:
        context += f"\n\nIMPORTANT: Use set_theme(\"{theme}\") early in your workflow."
    context += (
        "\n\nSTYLE GUIDE: Keep each section body to 2-3 concise sentences (under 40 words). "
        "Use bullet points or short punchy phrases instead of long paragraphs. "
        "White space is your friend — less text per slide is more impactful."
    )

    if is_bedrock:
        messages_bedrock = [{"role": "user", "content": [{"text": context}]}]
    else:
        messages_hf = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

    prev_quality = compute_quality_score(env.state)
    cumulative_reward = 0.0
    turns_used = 0
    steps = []
    t0 = time.time()

    for turn_idx in range(max_turns):
        try:
            if is_bedrock:
                assistant_text = _call_claude(
                    bedrock_client, messages_bedrock, SYSTEM_PROMPT,
                    bedrock_model_id, MAX_TOKENS_PER_TURN, 0.7)
            else:
                assistant_text = generate_finetuned(
                    model, tokenizer, messages_hf, max_new_tokens=512)
        except Exception as e:
            import traceback
            print(f"    [turn {turn_idx}] Error: {e}", flush=True)
            traceback.print_exc()
            break

        tool_call = extract_tool_call(assistant_text)
        if tool_call is None or "tool" not in tool_call:
            if is_bedrock:
                messages_bedrock.append({"role": "assistant", "content": [{"text": assistant_text}]})
                messages_bedrock.append({"role": "user", "content": [{"text": "You must respond with a JSON tool call."}]})
            else:
                messages_hf.append({"role": "assistant", "content": assistant_text})
                messages_hf.append({"role": "user", "content": "You must respond with a JSON tool call."})
            steps.append({
                "turn": turn_idx,
                "assistant_text": assistant_text,
                "tool_call": None,
                "observation": "No valid tool call extracted",
                "success": False,
            })
            continue

        params = tool_call.get("parameters", {})
        if not params:
            params = {k: v for k, v in tool_call.items() if k != "tool"}
        action = SlideForgeAction(tool=tool_call.get("tool", ""), parameters=params)
        obs = env.step(action)

        step_reward = compute_step_reward(
            state=env.state, prev_quality=prev_quality,
            tool=tool_call.get("tool", ""), success=obs.success)
        cumulative_reward += step_reward["step_reward"]
        prev_quality = step_reward["current_quality"]
        turns_used = turn_idx + 1

        if verbose:
            status = "OK" if obs.success else "FAIL"
            q = step_reward["current_quality"]["aggregate"]
            print(f"    [{model_name:<10}] turn {turn_idx:>2}: {tool_call['tool']:<20} "
                  f"{status:<4} | phase={obs.phase:<10} slides={obs.current_slide_count} | q={q:.3f}")

        turns_left = max_turns - turn_idx - 1
        obs_text = (f"Tool result (success={obs.success}):\n{obs.result}\n\n"
                    f"State: phase={obs.phase}, slides={obs.current_slide_count}/{brief.get('num_slides', 10)}, "
                    f"turns remaining={turns_left}")
        if obs.done:
            obs_text += "\n\nEpisode complete."
        elif turns_left <= 3:
            obs_text += f"\n\nURGENT: Only {turns_left} turn(s) left. Call finalize NOW."
        elif turns_left <= 6:
            obs_text += f"\n\nReminder: {turns_left} turns left. Wrap up and call finalize soon."

        steps.append({
            "turn": turn_idx,
            "assistant_text": assistant_text,
            "tool_call": tool_call,
            "observation": obs_text,
            "success": obs.success,
            "step_reward": step_reward["step_reward"],
            "quality": step_reward["current_quality"]["aggregate"],
            "phase": obs.phase,
            "slide_count": obs.current_slide_count,
        })

        if is_bedrock:
            messages_bedrock.append({"role": "assistant", "content": [{"text": assistant_text}]})
            messages_bedrock.append({"role": "user", "content": [{"text": obs_text}]})
        else:
            messages_hf.append({"role": "assistant", "content": assistant_text})
            messages_hf.append({"role": "user", "content": obs_text})

        if obs.done:
            break

    elapsed = time.time() - t0
    final_quality = compute_quality_score(env.state)

    return {
        "model": model_name,
        "brief": brief,
        "brief_topic": brief.get("topic", "?"),
        "completed": env.state.done and env.state.phase == "DONE",
        "turns_used": turns_used,
        "slides_created": sum(1 for h in env.state.slides_html if h),
        "final_quality": final_quality,
        "cumulative_reward": cumulative_reward,
        "elapsed_seconds": round(elapsed, 1),
        "theme": env.state.theme,
        "slides_html": env.state.slides_html,
        "outline": env.state.outline,
        "steps": steps,
    }


def _model_label(model_name: str) -> str:
    """Human-readable label for a model name."""
    if model_name == "finetuned":
        return "Fine-tuned (ours)"
    if model_name == "base":
        return "Base Qwen"
    if model_name in BEDROCK_MODELS:
        return BEDROCK_MODELS[model_name]["label"]
    return model_name


def print_comparison(results: list[dict]):
    all_model_names = []
    seen = set()
    for r in results:
        if r["model"] not in seen:
            all_model_names.append(r["model"])
            seen.add(r["model"])

    local_order = ["finetuned", "base"]
    active_models = [m for m in local_order if m in seen]
    active_models += [m for m in all_model_names if m not in local_order]

    print("\n" + "=" * 120)
    title_parts = [_model_label(m) for m in active_models]
    print(f"  EVALUATION: {' vs '.join(title_parts)}")
    print("=" * 120)

    col_w = max(14, max((len(m) for m in active_models), default=12) + 2)
    header = f"  {'Brief':<40} {'Model':<{col_w}} {'Done':<6} {'Turns':<7} {'Slides':<8} {'Quality':<9} {'Time':<8}"
    print(header)
    print("  " + "-" * 116)

    topics = []
    seen_topics = set()
    for r in results:
        if r["brief_topic"] not in seen_topics:
            topics.append(r["brief_topic"])
            seen_topics.add(r["brief_topic"])

    for topic in topics:
        short_topic = topic[:38] if len(topic) > 38 else topic
        for model_name in active_models:
            r = next((r for r in results if r["brief_topic"] == topic and r["model"] == model_name), None)
            if r:
                q = r["final_quality"].get("aggregate", 0.0)
                done_str = "YES" if r["completed"] else "NO"
                print(f"  {short_topic:<40} {model_name:<{col_w}} {done_str:<6} {r['turns_used']:<7} "
                      f"{r['slides_created']:<8} {q:<9.3f} {r['elapsed_seconds']:<8.1f}s")
        print()

    print("=" * 120)
    print("  AGGREGATE RESULTS")
    print("  " + "-" * 116)

    model_groups = {m: [r for r in results if r["model"] == m] for m in active_models}

    for model_name in active_models:
        group = model_groups[model_name]
        if not group:
            continue
        avg_q = sum(r["final_quality"].get("aggregate", 0) for r in group) / len(group)
        completed = sum(1 for r in group if r["completed"])
        avg_turns = sum(r["turns_used"] for r in group) / len(group)
        avg_slides = sum(r["slides_created"] for r in group) / len(group)
        avg_time = sum(r["elapsed_seconds"] for r in group) / len(group)

        sub_scores = {}
        for key in ["code_rules", "render_quality", "content_quality",
                     "claude_aesthetic_html", "claude_aesthetic_visual", "brief_reconstruction"]:
            vals = [r["final_quality"].get(key, 0) for r in group]
            sub_scores[key] = sum(vals) / len(vals) if vals else 0

        print(f"\n  {_model_label(model_name)}:")
        print(f"    Overall quality:  {avg_q:.3f}")
        print(f"    Completion rate:  {completed}/{len(group)} ({completed/len(group)*100:.0f}%)")
        print(f"    Avg turns used:   {avg_turns:.1f}")
        print(f"    Avg slides:       {avg_slides:.1f}")
        print(f"    Avg time/brief:   {avg_time:.1f}s")
        print(f"    Sub-scores:")
        for k, v in sub_scores.items():
            print(f"      {k:<28} {v:.3f}")

    def _avg_q(group):
        return sum(r["final_quality"].get("aggregate", 0) for r in group) / len(group) if group else 0

    ft_results = model_groups.get("finetuned", [])
    if ft_results and len(active_models) > 1:
        print(f"\n  {'─' * 60}")
        fq = _avg_q(ft_results)
        for other in active_models:
            if other == "finetuned":
                continue
            other_group = model_groups[other]
            if not other_group:
                continue
            cq = _avg_q(other_group)
            diff = fq - cq
            label = _model_label(other)
            print(f"  Quality delta (ours - {label}): {diff:+.3f}")
            if cq > 0:
                print(f"  Relative improvement vs {label}: {diff/cq*100:+.1f}%")

            wins, ties, losses = 0, 0, 0
            for topic in topics:
                ft = next((r for r in ft_results if r["brief_topic"] == topic), None)
                ot = next((r for r in other_group if r["brief_topic"] == topic), None)
                if ft and ot:
                    ft_q = ft["final_quality"].get("aggregate", 0)
                    ot_q = ot["final_quality"].get("aggregate", 0)
                    if ft_q > ot_q + 0.01:
                        wins += 1
                    elif ot_q > ft_q + 0.01:
                        losses += 1
                    else:
                        ties += 1
            if wins + ties + losses > 0:
                print(f"  Head-to-head vs {label}: {wins}W / {ties}T / {losses}L")
            print()

    print("=" * 120)


def main():
    bedrock_choices = list(BEDROCK_MODELS.keys())

    parser = argparse.ArgumentParser(
        description="Evaluate models on SlideForge presentations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available Bedrock models: {', '.join(bedrock_choices)}")
    parser.add_argument("--checkpoint", default="outputs/slideforge_grpo/checkpoint-200",
                        help="Path to fine-tuned LoRA checkpoint")
    parser.add_argument("--trajectories", default="outputs/edited_trajectories_2.json",
                        help="Training trajectories JSON (briefs are extracted from here)")
    parser.add_argument("--briefs-file", default=None,
                        help="Override: load briefs from this file instead of trajectories")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=35)
    parser.add_argument("--output-dir", default=OUTPUT_BASE)
    parser.add_argument("--skip-finetuned", action="store_true",
                        help="Skip the fine-tuned model")
    parser.add_argument("--include-base", action="store_true",
                        help="Also run the base (non-finetuned) Qwen model")
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-Coder-7B-Instruct",
                        help="Base model name for --include-base")
    parser.add_argument("--bedrock-models", nargs="*", default=["claude-opus"],
                        metavar="MODEL", choices=bedrock_choices,
                        help=f"Bedrock models to evaluate (default: claude-opus). "
                             f"Choices: {', '.join(bedrock_choices)}. "
                             f"Use --no-bedrock to skip all Bedrock models.")
    parser.add_argument("--no-bedrock", action="store_true",
                        help="Skip all Bedrock model evaluations")
    # Keep --skip-claude as a hidden alias for backward compat
    parser.add_argument("--skip-claude", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.skip_claude:
        args.no_bedrock = True

    active_bedrock = [] if args.no_bedrock else (args.bedrock_models or [])

    if args.briefs_file:
        with open(args.briefs_file) as f:
            briefs = json.load(f)
    else:
        with open(args.trajectories) as f:
            trajectories = json.load(f)
        completed = [t for t in trajectories if t.get("completed", False)]
        briefs = [t["brief"] for t in completed]
        print(f"Loaded {len(briefs)} briefs from training trajectories")

    if args.limit:
        briefs = briefs[:args.limit]
    print(f"Evaluating on {len(briefs)} briefs\n")

    ft_model, ft_tokenizer = None, None
    if not args.skip_finetuned:
        print(f"Loading fine-tuned model from {args.checkpoint}...")
        ft_model, ft_tokenizer = load_finetuned_model(args.checkpoint)
        print("Fine-tuned model loaded.\n")

    base_model, base_tokenizer = None, None
    if args.include_base:
        print(f"Loading base model from {args.base_model}...")
        base_model, base_tokenizer = load_finetuned_model(args.base_model)
        print("Base model loaded.\n")

    bedrock_client = None
    if active_bedrock:
        bedrock_client = _create_bedrock_client()
        print(f"Bedrock client ready. Models: {', '.join(active_bedrock)}\n")

    ft_dir = f"{args.output_dir}/finetuned"
    base_dir = f"{args.output_dir}/base"
    if not args.skip_finetuned:
        os.makedirs(ft_dir, exist_ok=True)
    if args.include_base:
        os.makedirs(base_dir, exist_ok=True)
    for bm in active_bedrock:
        os.makedirs(f"{args.output_dir}/{bm}", exist_ok=True)

    all_results = []

    def _try_load_cached(deck_dir: str, model_name: str, brief: dict) -> dict | None:
        """Load a previous run's result from disk if quality.json exists.

        Skips cache (returns None) when quality=0 and turns=0, which
        indicates the episode failed before any turn ran (e.g. expired
        API key) rather than a genuine zero-quality result.
        """
        qpath = f"{deck_dir}/quality.json"
        tpath = f"{deck_dir}/trajectory.json"
        if not os.path.exists(qpath):
            return None
        try:
            with open(qpath) as f:
                quality = json.load(f)
            meta = {}
            if os.path.exists(tpath):
                with open(tpath) as f:
                    meta = json.load(f)
            turns = meta.get("turns_used", 0) or len(meta.get("steps", []))
            if quality.get("aggregate", 0) == 0.0 and turns == 0:
                return None
            return {
                "model": model_name,
                "brief": brief,
                "brief_topic": brief.get("topic", "?"),
                "completed": meta.get("completed", False),
                "turns_used": meta.get("turns_used", 0),
                "slides_created": meta.get("slides_created",
                    sum(1 for s in Path(deck_dir, "slides").glob("*.html")) if Path(deck_dir, "slides").exists() else 0),
                "final_quality": quality,
                "cumulative_reward": meta.get("cumulative_reward", 0),
                "elapsed_seconds": meta.get("elapsed_seconds", 0),
                "theme": meta.get("theme", ""),
                "slides_html": [],
                "outline": [],
                "steps": meta.get("steps", []),
            }
        except Exception:
            return None

    for i, brief in enumerate(briefs):
        brief = dict(brief)
        brief["preferred_theme"] = "creative"
        topic = brief.get("topic", "?")
        slug = _slug(topic)
        print(f"{'='*80}")
        print(f"[{i+1}/{len(briefs)}] {topic}")
        print(f"{'='*80}")

        if not args.skip_finetuned:
            deck_dir = f"{ft_dir}/{slug}"
            cached = _try_load_cached(deck_dir, "finetuned", brief)
            if cached:
                all_results.append(cached)
                q = cached["final_quality"].get("aggregate", 0)
                print(f"\n  >> Fine-tuned model: [cached] q={q:.3f}")
            else:
                print(f"\n  >> Fine-tuned model:")
                r = run_episode("finetuned", brief, args.max_turns,
                                model=ft_model, tokenizer=ft_tokenizer, verbose=True)
                all_results.append(r)
                q = r["final_quality"].get("aggregate", 0)
                print(f"  => q={q:.3f} | turns={r['turns_used']} | "
                      f"slides={r['slides_created']} | done={r['completed']}")
                save_deck(r, deck_dir)

        if args.include_base:
            deck_dir = f"{base_dir}/{slug}"
            cached = _try_load_cached(deck_dir, "base", brief)
            if cached:
                all_results.append(cached)
                q = cached["final_quality"].get("aggregate", 0)
                print(f"\n  >> Base Qwen (no fine-tuning): [cached] q={q:.3f}")
            else:
                print(f"\n  >> Base Qwen (no fine-tuning):")
                r = run_episode("base", brief, args.max_turns,
                                model=base_model, tokenizer=base_tokenizer, verbose=True)
                all_results.append(r)
                q = r["final_quality"].get("aggregate", 0)
                print(f"  => q={q:.3f} | turns={r['turns_used']} | "
                      f"slides={r['slides_created']} | done={r['completed']}")
                save_deck(r, deck_dir)

        for bm in active_bedrock:
            bm_info = BEDROCK_MODELS[bm]
            deck_dir = f"{args.output_dir}/{bm}/{slug}"
            cached = _try_load_cached(deck_dir, bm, brief)
            if cached:
                all_results.append(cached)
                q = cached["final_quality"].get("aggregate", 0)
                print(f"\n  >> {bm_info['label']}: [cached] q={q:.3f}")
            else:
                print(f"\n  >> {bm_info['label']}:")
                r = run_episode(bm, brief, args.max_turns,
                                bedrock_client=bedrock_client,
                                bedrock_model_id=bm_info["model_id"],
                                verbose=True)
                all_results.append(r)
                q = r["final_quality"].get("aggregate", 0)
                print(f"  => q={q:.3f} | turns={r['turns_used']} | "
                      f"slides={r['slides_created']} | done={r['completed']}")
                save_deck(r, deck_dir)

    print_comparison(all_results)

    results_file = f"{args.output_dir}/results.json"
    existing = []
    if os.path.exists(results_file):
        with open(results_file) as f:
            existing = json.load(f)

    current_models = {r["model"] for r in all_results}
    kept = [r for r in existing if r["model"] not in current_models]

    serializable = []
    for r in all_results:
        serializable.append({k: v for k, v in r.items() if k not in ["slides_html"]})

    merged = kept + serializable
    with open(results_file, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"\nFull results saved to {results_file} "
          f"({len(kept)} kept + {len(serializable)} new/updated = {len(merged)} total)")
    if not args.skip_finetuned:
        print(f"Fine-tuned decks:  {ft_dir}/")
    if args.include_base:
        print(f"Base Qwen decks:   {base_dir}/")
    for bm in active_bedrock:
        print(f"{BEDROCK_MODELS[bm]['label']:.<20s} {args.output_dir}/{bm}/")


if __name__ == "__main__":
    main()
