"""Compute aggregate results across all models in results.json.

Usage:
    python -m training.compute_aggregate \
        --eval-results outputs/output_all48/results.json \
        --output outputs/output_all48/aggregate_results.json
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

SUB_KEYS = [
    "code_rules", "render_quality", "content_quality",
    "claude_aesthetic_html", "claude_aesthetic_visual", "brief_reconstruction",
]

MODEL_LABELS = {
    "finetuned": "Fine-tuned (ours)",
    "base": "Base Qwen",
    "claude-opus": "Claude Opus 4.6",
    "claude-sonnet": "Claude Sonnet 4.6",
    "gpt-oss-120b": "GPT OSS 120B",
    "llama4-scout": "Llama 4 Scout 17B",
    "claude": "Claude Opus (trajectories)",
}


def compute_model_aggregate(group: list[dict]) -> dict:
    n = len(group)
    if n == 0:
        return {}

    avg_q = sum(r["final_quality"].get("aggregate", 0) for r in group) / n
    strict_completed = sum(1 for r in group if r.get("completed"))
    effective_completed = sum(1 for r in group if r.get("completed") or r.get("slides_created", 0) > 0)
    avg_turns = sum(r.get("turns_used", 0) for r in group) / n
    avg_slides = sum(r.get("slides_created", 0) for r in group) / n
    avg_time = sum(r.get("elapsed_seconds", 0) for r in group) / n

    sub_scores = {}
    for key in SUB_KEYS:
        vals = [r["final_quality"].get(key, 0) for r in group]
        sub_scores[key] = round(sum(vals) / len(vals), 4)

    return {
        "overall_quality": round(avg_q, 4),
        "completion_rate": f"{effective_completed}/{n} ({effective_completed / n * 100:.1f}%)",
        "completion_rate_strict": f"{strict_completed}/{n} ({strict_completed / n * 100:.1f}%)",
        "completed": effective_completed,
        "completed_strict": strict_completed,
        "total_briefs": n,
        "avg_turns_used": round(avg_turns, 1),
        "avg_slides_created": round(avg_slides, 1),
        "avg_time_per_brief_seconds": round(avg_time, 1),
        "sub_scores": sub_scores,
    }


def compute_head_to_head(a_results: list[dict], b_results: list[dict]) -> dict:
    wins, ties, losses = 0, 0, 0
    per_brief = []

    b_by_topic = {r["brief_topic"]: r for r in b_results}
    matched = 0
    for a in a_results:
        b = b_by_topic.get(a["brief_topic"])
        if not b:
            continue
        matched += 1
        aq = a["final_quality"].get("aggregate", 0)
        bq = b["final_quality"].get("aggregate", 0)
        delta = aq - bq

        if delta > 0.01:
            wins += 1
            result = "win"
        elif delta < -0.01:
            losses += 1
            result = "loss"
        else:
            ties += 1
            result = "tie"

        per_brief.append({
            "topic": a["brief_topic"],
            "a_quality": round(aq, 4),
            "b_quality": round(bq, 4),
            "delta": round(delta, 4),
            "result": result,
        })

    if matched == 0:
        return {"quality_delta": 0, "relative_improvement_pct": None,
                "head_to_head": {"wins": 0, "ties": 0, "losses": 0}, "per_brief": []}

    a_avg = sum(p["a_quality"] for p in per_brief) / matched
    b_avg = sum(p["b_quality"] for p in per_brief) / matched
    quality_delta = round(a_avg - b_avg, 4)

    return {
        "quality_delta": quality_delta,
        "relative_improvement_pct": round(quality_delta / b_avg * 100, 1) if b_avg > 0 else None,
        "head_to_head": {"wins": wins, "ties": ties, "losses": losses},
        "per_brief": sorted(per_brief, key=lambda x: x["delta"], reverse=True),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute aggregate results across all models")
    parser.add_argument("--eval-results", default="outputs/output_all48/results.json")
    parser.add_argument("--output", default="outputs/output_all48/aggregate_results.json")
    args = parser.parse_args()

    with open(args.eval_results) as f:
        all_results = json.load(f)

    by_model = defaultdict(list)
    for r in all_results:
        by_model[r["model"]].append(r)

    model_names = list(by_model.keys())
    print(f"Models found: {', '.join(f'{m} ({len(by_model[m])})' for m in model_names)}\n")

    aggregate = {m: compute_model_aggregate(by_model[m]) for m in model_names}

    comparisons = {}
    for a, b in combinations(model_names, 2):
        comparisons[f"{a}_vs_{b}"] = compute_head_to_head(by_model[a], by_model[b])

    output = {"aggregate": aggregate, "comparisons": comparisons}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print("=" * 80)
    print(f"  AGGREGATE RESULTS ({len(model_names)} models)")
    print("=" * 80)

    for m in model_names:
        a = aggregate[m]
        label = MODEL_LABELS.get(m, m)
        print(f"\n  {label}:")
        print(f"    Overall quality:  {a['overall_quality']:.4f}")
        print(f"    Completion rate:  {a['completion_rate']}  (strict: {a['completion_rate_strict']})")
        print(f"    Avg turns used:   {a['avg_turns_used']}")
        print(f"    Avg slides:       {a['avg_slides_created']}")
        print(f"    Avg time/brief:   {a['avg_time_per_brief_seconds']}s")
        print(f"    Sub-scores:")
        for k, v in a["sub_scores"].items():
            print(f"      {k:<28} {v:.4f}")

    print(f"\n{'=' * 80}")
    print(f"  PAIRWISE COMPARISONS (finetuned vs others)")
    print(f"{'=' * 80}")

    for key, comp in comparisons.items():
        if not key.startswith("finetuned_vs_"):
            continue
        h = comp["head_to_head"]
        other = key.replace("finetuned_vs_", "")
        label = MODEL_LABELS.get(other, other)
        print(f"\n  Fine-tuned vs {label}:")
        print(f"    Quality delta:        {comp['quality_delta']:+.4f}")
        if comp["relative_improvement_pct"] is not None:
            print(f"    Relative improvement: {comp['relative_improvement_pct']:+.1f}%")
        print(f"    Head-to-head:         {h['wins']}W / {h['ties']}T / {h['losses']}L")

    print(f"\n{'=' * 80}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
