#!/usr/bin/env python3
"""Reduced local reproduction of the AQuA ICL pipeline.

This script intentionally bypasses the unavailable Ollama runtime, the custom
`ltr` package, and the truncated precomputed AQuA embeddings. It keeps the
core ICL chain: candidate exemplar subsets -> local LLM generation -> BERTScore
reward -> best-subset selection -> held-out AQuA accuracy.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import multiprocessing as mp
import queue as queue_module
import random
import re
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from bert_score import BERTScorer
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
LABELS = ("A", "B", "C", "D", "E")
SYSTEM_PROMPT = (
    "You are a careful multiple-choice math tutor. Follow the given examples. "
    "For each question, output exactly two lines: "
    "'Final Answer: <letter>' followed by 'Reason: <one short sentence>'."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/home/model/Qwen2.5-7B-Instruct",
        help="Local causal language model directory.",
    )
    parser.add_argument(
        "--scorer-model",
        default="/home/model/roberta-large",
        help="Local BERTScore model directory.",
    )
    parser.add_argument("--train-pool-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=8)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--test-size", type=int, default=-1, help="-1 uses all remaining dev rows.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output" / "aqua_local_reproduction"),
    )
    return parser.parse_args()


def clean_text(value: Any, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:limit] if limit is not None else text


def format_options(raw_options: Any) -> str:
    options = ast.literal_eval(raw_options) if isinstance(raw_options, str) else raw_options
    if not isinstance(options, (list, tuple)):
        raise ValueError(f"Unsupported options value: {raw_options!r}")
    return "\n".join(str(option).strip() for option in options)


def build_prompt(target: pd.Series, exemplars: pd.DataFrame) -> str:
    parts = [
        "Solve the following AQuA-RAT multiple-choice math problems.",
        "Use the examples as in-context demonstrations. Do not add extra commentary.",
        "",
    ]
    for _, example in exemplars.iterrows():
        parts.extend(
            [
                f"Question: {clean_text(example['question'])}",
                f"Options:\n{format_options(example['options'])}",
                "Response:",
                f"Final Answer: {str(example['correct']).strip().upper()}",
                f"Reason: {clean_text(example['rationale'], 240)}",
                "",
            ]
        )
    parts.extend(
        [
            f"Question: {clean_text(target['question'])}",
            f"Options:\n{format_options(target['options'])}",
            "Response:",
        ]
    )
    return "\n".join(parts)


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = ROOT / "data" / "AQUA_RAT" / "aquarat_train.csv"
    dev_path = ROOT / "data" / "AQUA_RAT" / "aquarat_dev.csv"
    train = pd.read_csv(train_path)
    dev = pd.read_csv(dev_path)
    required = {"question", "options", "rationale", "correct"}
    for name, frame in (("train", train), ("dev", dev)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        frame = frame.dropna(subset=list(required))
        frame["correct"] = frame["correct"].astype(str).str.strip().str.upper()
        frame = frame[frame["correct"].isin(LABELS)]
        if name == "train":
            train = frame.reset_index(drop=True)
        else:
            dev = frame.reset_index(drop=True)
    return train, dev


def stratified_subset(pool: pd.DataFrame, size: int, rng: random.Random) -> list[int]:
    if size > len(pool):
        raise ValueError(f"Cannot sample {size} exemplars from a pool of {len(pool)}")

    chosen: list[int] = []
    groups = [group.index.tolist() for _, group in pool.groupby("correct", sort=True)]
    rng.shuffle(groups)
    for group in groups:
        if len(chosen) >= size:
            break
        rng.shuffle(group)
        chosen.append(group.pop())

    remaining = [idx for idx in pool.index.tolist() if idx not in set(chosen)]
    rng.shuffle(remaining)
    chosen.extend(remaining[: size - len(chosen)])
    rng.shuffle(chosen)
    return chosen


def extract_answer(text: str) -> str | None:
    matches = re.findall(r"final\s*answer\s*[:\-]\s*\(?\s*([A-E])\b", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    tail = text.strip()[-80:]
    matches = re.findall(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", tail)
    return matches[-1].upper() if matches else None


def load_generator(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to("cuda")
    model.eval()
    return model, tokenizer


def release_generator() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(chat_text, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _bertscore_worker(
    scorer_model: str,
    predictions: list[str],
    references: list[str],
    result_queue,
) -> None:
    try:
        scorer = BERTScorer(
            model_type=scorer_model,
            num_layers=17,
            batch_size=4,
            device="cuda",
            rescale_with_baseline=False,
        )
        precision, recall, f1 = scorer.score(predictions, references)
        result_queue.put(
            {
                "ok": True,
                "precision": precision.float().cpu().tolist(),
                "recall": recall.float().cpu().tolist(),
                "f1": f1.float().cpu().tolist(),
            }
        )
    except Exception:
        result_queue.put({"ok": False, "error": traceback.format_exc()})


def score_with_bertscore(
    scorer_model: str, predictions: list[str], references: list[str]
) -> dict[str, list[float]]:
    """Run BERTScore in a child process so the CUDA context is fully released."""
    torch.cuda.empty_cache()
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_bertscore_worker,
        args=(scorer_model, predictions, references, result_queue),
        daemon=True,
    )
    process.start()
    try:
        result = result_queue.get(timeout=900)
    except queue_module.Empty as exc:
        process.terminate()
        process.join(timeout=30)
        raise RuntimeError("BERTScore child process timed out") from exc
    process.join(timeout=120)
    if process.is_alive():
        process.terminate()
        process.join(timeout=30)
    if not result.get("ok"):
        raise RuntimeError(f"BERTScore child process failed:\n{result.get('error')}")
    return {
        "precision": result["precision"],
        "recall": result["recall"],
        "f1": result["f1"],
    }

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train, dev = load_dataset()
    train = train.sample(n=min(args.train_pool_size, len(train)), random_state=args.seed)
    train = train.reset_index(drop=True)
    if args.val_size + args.shots > len(train):
        raise ValueError("Validation + exemplar requirements exceed the sampled train pool")

    val = train.iloc[: args.val_size].copy()
    exemplar_pool = train.iloc[args.val_size :].copy()
    test = dev.copy()
    if args.test_size >= 0:
        test = test.iloc[: args.test_size].copy()

    rng = random.Random(args.seed)
    candidate_indices = [
        stratified_subset(exemplar_pool, args.shots, rng)
        for _ in range(args.candidate_count)
    ]

    print(
        f"Pipeline: candidates={args.candidate_count}, shots={args.shots}, "
        f"val={len(val)}, test={len(test)}",
        flush=True,
    )
    print(f"Generator: {args.model_path}", flush=True)
    print(f"Scorer: {args.scorer_model}", flush=True)

    model, tokenizer = load_generator(args.model_path)
    candidate_records: list[dict[str, Any]] = []
    for candidate_id, indices in enumerate(candidate_indices, start=1):
        exemplars = exemplar_pool.loc[indices]
        print(f"Generating validation outputs for candidate {candidate_id}/{len(candidate_indices)}", flush=True)
        predictions: list[str] = []
        references: list[str] = []
        parsed_answers: list[str | None] = []
        for row_number, (_, row) in enumerate(val.iterrows(), start=1):
            prompt = build_prompt(row, exemplars)
            response = generate(model, tokenizer, prompt, args.max_new_tokens)
            answer = extract_answer(response)
            predictions.append(response)
            references.append(clean_text(row["rationale"]))
            parsed_answers.append(answer)
            print(
                f"  candidate={candidate_id} val={row_number}/{len(val)} "
                f"pred={answer} gt={row['correct']}",
                flush=True,
            )
        candidate_records.append(
            {
                "candidate_id": candidate_id,
                "indices": indices,
                "predictions": predictions,
                "references": references,
                "answers": parsed_answers,
            }
        )
    del model, tokenizer
    release_generator()

    print("Scoring validation outputs with local roberta-large BERTScore", flush=True)
    all_predictions = [p for record in candidate_records for p in record["predictions"]]
    all_references = [r for record in candidate_records for r in record["references"]]
    bert = score_with_bertscore(args.scorer_model, all_predictions, all_references)

    cursor = 0
    for record in candidate_records:
        count = len(record["predictions"])
        f1_scores = bert["f1"][cursor : cursor + count]
        accuracy = sum(
            answer == str(gt).strip().upper()
            for answer, gt in zip(record["answers"], val["correct"].tolist())
        ) / count
        record["bertscore_f1"] = f1_scores
        record["mean_bertscore_f1"] = sum(f1_scores) / count
        record["validation_accuracy"] = accuracy
        cursor += count

    best = max(candidate_records, key=lambda record: record["mean_bertscore_f1"])
    print(
        f"Selected candidate {best['candidate_id']} with "
        f"BERTScore F1={best['mean_bertscore_f1']:.4f}, "
        f"val accuracy={best['validation_accuracy']:.4f}",
        flush=True,
    )

    model, tokenizer = load_generator(args.model_path)
    selected_exemplars = exemplar_pool.loc[best["indices"]]
    test_records: list[dict[str, Any]] = []
    matches = 0
    for row_number, (_, row) in enumerate(test.iterrows(), start=1):
        prompt = build_prompt(row, selected_exemplars)
        response = generate(model, tokenizer, prompt, args.max_new_tokens)
        answer = extract_answer(response)
        ground_truth = str(row["correct"]).strip().upper()
        is_correct = answer == ground_truth
        matches += int(is_correct)
        test_records.append(
            {
                "row": row_number,
                "question": row["question"],
                "ground_truth": ground_truth,
                "prediction": answer,
                "correct": is_correct,
                "response": response,
            }
        )
        if row_number == 1 or row_number % 10 == 0 or row_number == len(test):
            print(
                f"  test={row_number}/{len(test)} pred={answer} gt={ground_truth} "
                f"running_accuracy={matches / row_number:.4f}",
                flush=True,
            )
    del model, tokenizer
    release_generator()

    accuracy = matches / len(test) if len(test) else 0.0
    elapsed = time.time() - started

    with (output_dir / "candidate_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "indices",
                "mean_bertscore_f1",
                "validation_accuracy",
            ],
        )
        writer.writeheader()
        for record in candidate_records:
            writer.writerow(
                {
                    "candidate_id": record["candidate_id"],
                    "indices": json.dumps(record["indices"]),
                    "mean_bertscore_f1": record["mean_bertscore_f1"],
                    "validation_accuracy": record["validation_accuracy"],
                }
            )

    pd.DataFrame(test_records).to_csv(output_dir / "test_predictions.csv", index=False)
    validation_details = []
    for record in candidate_records:
        for row, answer, response, f1 in zip(
            val.to_dict(orient="records"),
            record["answers"],
            record["predictions"],
            record["bertscore_f1"],
        ):
            validation_details.append(
                {
                    "candidate_id": record["candidate_id"],
                    "question": row["question"],
                    "ground_truth": row["correct"],
                    "prediction": answer,
                    "bertscore_f1": f1,
                    "response": response,
                }
            )
    pd.DataFrame(validation_details).to_csv(
        output_dir / "validation_generations.csv", index=False
    )

    exemplar_details = selected_exemplars[
        ["question", "options", "rationale", "correct"]
    ].to_dict(orient="records")
    metrics = {
        "status": "completed",
        "method": (
            "Reduced AQuA ICL: stratified candidate exemplar subsets; local causal LM "
            "generation; local roberta-large BERTScore reward; best subset evaluated "
            "on held-out AQuA dev."
        ),
        "generator_model": args.model_path,
        "scorer_model": args.scorer_model,
        "seed": args.seed,
        "train_pool_size": len(train),
        "validation_size": len(val),
        "test_size": len(test),
        "candidate_count": args.candidate_count,
        "shots": args.shots,
        "selected_candidate": best["candidate_id"],
        "selected_indices": best["indices"],
        "selected_validation_bertscore_f1": best["mean_bertscore_f1"],
        "selected_validation_accuracy": best["validation_accuracy"],
        "test_matches": matches,
        "test_accuracy": accuracy,
        "elapsed_seconds": elapsed,
        "selected_exemplars": exemplar_details,
        "bypasses": [
            "Ollama runtime replaced with local Transformers generation",
            "custom ltr/RankNet training bypassed with validation-reward subset selection",
            "truncated precomputed AQuA embeddings bypassed with direct tokenized prompts",
        ],
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
