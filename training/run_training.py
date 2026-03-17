"""Main training script for SlideForge GRPO.

Loads pre-collected trajectories from rollouts.py and trains
via GRPO using unsloth + LoRA + 4-bit quantization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from training.grpo_trainer import create_trainer, slideforge_reward
from training.rollouts import trajectories_to_dataset


def load_trajectories(path: str, min_quality: float = 0.0) -> list[dict]:
    """Load trajectory JSON, keep only completed episodes."""
    with open(path) as f:
        trajectories = json.load(f)

    completed = [t for t in trajectories if t.get("completed", False)]

    if min_quality > 0:
        before = len(completed)
        completed = [
            t for t in completed
            if t.get("final_quality", {}).get("aggregate", 0.0) >= min_quality
        ]
        print(f"Quality filter ({min_quality}): {before} -> {len(completed)}")

    print(f"Loaded {len(completed)} completed trajectories from {path}")
    return completed


def main():
    parser = argparse.ArgumentParser(description="SlideForge GRPO Training")
    parser.add_argument("--trajectories", required=True,
                        help="Path to trajectory JSON from rollouts.py")
    parser.add_argument("--min-quality", type=float, default=0.0)
    parser.add_argument("--model", default="unsloth/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/slideforge_grpo")
    args = parser.parse_args()

    trajectories = load_trajectories(args.trajectories, args.min_quality)
    if not trajectories:
        print("No trajectories to train on. Run rollouts first.")
        return

    print("Building GRPO dataset from trajectories...")
    dataset = trajectories_to_dataset(trajectories, format="grpo")
    print(f"Dataset: {len(dataset)} step-level rows")

    print("Creating trainer...")
    from trl import GRPOTrainer

    model, tokenizer, training_args = create_trainer(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
    )
    training_args.output_dir = args.output_dir

    max_prompt_tokens = args.max_seq_length - training_args.max_completion_length

    def _to_msgs(raw):
        """HF datasets stores list-of-dicts in columnar format; convert back."""
        if isinstance(raw, dict) and "role" in raw:
            return [{"role": r, "content": c}
                    for r, c in zip(raw["role"], raw["content"])]
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            return [dict(m) for m in raw]
        return raw

    def _msgs_to_text(msgs):
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

    def _count_tokens(text):
        return len(tokenizer.encode(text))

    raw_prompts = [_to_msgs(r["prompt"]) for r in dataset]
    formatted = []
    for msgs in raw_prompts:
        text = _msgs_to_text(msgs)
        n = _count_tokens(text)
        if n > max_prompt_tokens:
            head = [m for m in msgs[:2] if m.get("role") in ("system", "user")]
            tail = list(msgs[2:])
            while tail:
                text = _msgs_to_text(head + tail)
                if _count_tokens(text) <= max_prompt_tokens:
                    break
                tail = tail[2:] if len(tail) >= 2 else tail[1:]
            text = _msgs_to_text(head + tail)
            n = _count_tokens(text)
        if n > max_prompt_tokens:
            ids = tokenizer.encode(text)[:max_prompt_tokens]
            text = tokenizer.decode(ids, skip_special_tokens=False)
            n = _count_tokens(text)
        formatted.append(text)

    from datasets import Dataset as HFDataset
    prompt_dataset = HFDataset.from_dict({"prompt": formatted})
    lengths = [_count_tokens(t) for t in formatted]
    print(f"Prompts: {len(formatted)} rows | max {max(lengths)} tokens | limit {max_prompt_tokens}")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[slideforge_reward],
        args=training_args,
        train_dataset=prompt_dataset,
    )

    print("Starting GRPO training...")
    trainer.train()
    print("Training complete!")

    save_path = f"{args.output_dir}/final"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
