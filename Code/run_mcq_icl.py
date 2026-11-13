#!/usr/bin/env python3
"""DearICL 4-shot evaluation for ARC-Challenge and OpenBookQA."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList


ROOT = Path(__file__).resolve().parents[1]
LETTERS = "ABCDE"
SYSTEM_PROMPT = (
    "You are a careful multiple-choice science assistant. "
    "Use the examples as in-context demonstrations. "
    "Return only the single correct option letter and do not explain."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("arc_challenge", "openbookqa"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--method", default="dearicl_4shot", choices=("dearicl_4shot",))
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--embedding-model", default="/home/model/roberta-large")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank-epochs", type=int, default=100)
    parser.add_argument("--rank-patience", type=int, default=10)
    parser.add_argument("--rank-batch-size", type=int, default=16)
    parser.add_argument("--limit-validation", type=int, default=0, help="Debug only; 0 uses configured size.")
    parser.add_argument("--limit-test", type=int, default=0, help="Debug only; 0 uses the full test split.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_path(root: Path, name: str) -> Path:
    return root / ("ARC_CHALLENGE" if name == "arc_challenge" else "OPENBOOKQA")


def load_local_dataset(root: Path, name: str) -> tuple[DatasetDict, str]:
    path = dataset_path(root, name)
    if not path.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {path}. Download ARC_CHALLENGE and OPENBOOKQA first."
        )
    ds = load_from_disk(str(path))
    question_key = "question" if "question" in ds["train"].column_names else "question_stem"
    return ds, question_key


def normalize_example(row: dict[str, Any], question_key: str) -> dict[str, Any]:
    raw_labels = [str(label) for label in row["choices"]["label"]]
    texts = [str(text).strip() for text in row["choices"]["text"]]
    if len(raw_labels) != len(texts) or not 3 <= len(texts) <= 5:
        raise ValueError(f"Unsupported choices for {row['id']}: {row['choices']}")
    labels = list(LETTERS[: len(texts)])
    answer_key = str(row["answerKey"])
    if answer_key not in raw_labels:
        raise ValueError(f"answerKey {answer_key!r} not in {raw_labels!r} for {row['id']}")
    answer = labels[raw_labels.index(answer_key)]
    return {
        "id": str(row["id"]),
        "question": str(row[question_key]).strip(),
        "choices": texts,
        "labels": labels,
        "answer": answer,
    }


def normalized_rows(ds: DatasetDict, split: str, question_key: str) -> list[dict[str, Any]]:
    return [normalize_example(row, question_key) for row in ds[split]]


def format_question(example: dict[str, Any], include_answer: bool) -> str:
    options = "\n".join(
        f"{label}. {text}" for label, text in zip(example["labels"], example["choices"])
    )
    text = f"Question: {example['question']}\nOptions:\n{options}\nAnswer:"
    if include_answer:
        text += f" {example['answer']}"
    return text


def build_prompt(target: dict[str, Any], exemplars: list[dict[str, Any]], tokenizer, chat_model: bool) -> str:
    demonstrations = "\n\n".join(format_question(example, include_answer=True) for example in exemplars)
    body = (
        "Solve each multiple-choice science question. "
        "Use the following examples as demonstrations.\n\n"
        f"{demonstrations}\n\n"
        "Now answer the final question. Output only one option letter.\n\n"
        f"{format_question(target, include_answer=False)}"
    )
    if chat_model:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": body},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return body + "\n"


def extract_answer(output: str, valid_labels: str = LETTERS) -> str | None:
    text = output.strip().upper()
    if not text:
        return None
    patterns = [
        r"(?:FINAL\s+ANSWER|ANSWER)\s*(?:IS|:)?\s*[\(\[]?([A-E])\b",
        r"^\s*[\(\[]?([A-E])\b",
        r"\b([A-E])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        distinct = {match for match in matches if match in valid_labels}
        if len(distinct) == 1:
            return distinct.pop()
        if len(distinct) > 1:
            return None
    return None


def load_embedding_model(path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def embed_texts(texts: list[str], tokenizer, model, device: torch.device, batch_size: int = 32) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        output = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(output.dtype)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
        vectors.append(pooled.cpu().numpy())
    return np.concatenate(vectors, axis=0).astype(np.float32)


def question_embedding_text(example: dict[str, Any]) -> str:
    options = " ".join(f"{label}. {text}" for label, text in zip(example["labels"], example["choices"]))
    return f"{example['question']} {options}"


def stratified_indices(labels: list[str], size: int, seed: int) -> list[int]:
    """Stratified sample that also supports classes represented by a single item."""
    if size > len(labels):
        raise ValueError(f"Requested validation size {size} > dataset size {len(labels)}")
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[label].append(index)
    if size < len(by_label):
        raise ValueError(
            f"Validation size {size} is too small for {len(by_label)} answer classes"
        )
    counts = {label: len(indices) for label, indices in by_label.items()}
    allocations = {label: max(1, int(size * count / len(labels))) for label, count in counts.items()}
    while sum(allocations.values()) > size:
        label = max(allocations, key=lambda item: allocations[item] - size * counts[item] / len(labels))
        if allocations[label] == 1:
            break
        allocations[label] -= 1
    while sum(allocations.values()) < size:
        label = max(
            (label for label in counts if allocations[label] < counts[label]),
            key=lambda item: size * counts[item] / len(labels) - allocations[item],
        )
        allocations[label] += 1
    rng = random.Random(seed)
    selected: list[int] = []
    for label in sorted(by_label):
        indices = by_label[label][:]
        rng.shuffle(indices)
        selected.extend(indices[: allocations[label]])
    return sorted(selected)


def build_candidates(
    train_embeddings: np.ndarray,
    shots: int,
    candidate_count: int,
    cluster_count: int,
    seed: int,
) -> list[list[int]]:
    if shots != cluster_count:
        raise ValueError("This experiment expects one exemplar per cluster and shots == clusters")
    labels = KMeans(n_clusters=cluster_count, random_state=seed, n_init=10).fit_predict(train_embeddings)
    groups = [np.flatnonzero(labels == cluster_id) for cluster_id in range(cluster_count)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("KMeans produced an empty cluster")
    rng = np.random.default_rng(seed)
    candidates: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    while len(candidates) < candidate_count:
        picks = tuple(sorted(int(rng.choice(group)) for group in groups))
        attempts += 1
        if picks not in seen:
            seen.add(picks)
            candidates.append(list(picks))
        if attempts > candidate_count * 100:
            raise RuntimeError("Unable to generate distinct candidate subsets")
    return candidates


def load_generator(path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    model.generation_config.max_length = None
    model.eval()
    return tokenizer, model


class OptionLetterProcessor(LogitsProcessor):
    """Force one option token, then terminate generation with EOS."""

    def __init__(self, option_token_ids: list[int], eos_token_id: int) -> None:
        self.option_token_ids = option_token_ids
        self.eos_token_id = eos_token_id
        self.step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        masked = torch.full_like(scores, -float("inf"))
        if self.step == 0:
            masked[:, self.option_token_ids] = scores[:, self.option_token_ids]
        else:
            masked[:, self.eos_token_id] = scores[:, self.eos_token_id]
        self.step += 1
        return masked


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    max_input_tokens: int,
    valid_labels: str,
) -> tuple[str, str | None]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)
    option_token_ids: list[int] = []
    for label in valid_labels:
        for surface in (label, f" {label}"):
            token_ids = tokenizer.encode(surface, add_special_tokens=False)
            if len(token_ids) == 1 and token_ids[0] not in option_token_ids:
                option_token_ids.append(token_ids[0])
    processors = LogitsProcessorList(
        [OptionLetterProcessor(option_token_ids, tokenizer.eos_token_id)]
    )
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        logits_processor=processors,
    )
    generated = tokenizer.decode(
        output[0, encoded["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return generated.strip(), extract_answer(generated, valid_labels)


class RankNet(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def candidate_feature(query_vector: np.ndarray, exemplar_vectors: np.ndarray) -> np.ndarray:
    return np.vstack([query_vector[None, :], exemplar_vectors]).mean(axis=0).astype(np.float32)


def build_ranking_pairs(
    validation_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    candidate_indices: list[list[int]],
    feedback: dict[int, list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left, right, targets, query_ids = [], [], [], []
    for query_index in range(len(validation_embeddings)):
        rewards = [feedback[candidate_id][query_index] for candidate_id in range(len(candidate_indices))]
        for i in range(len(candidate_indices)):
            for j in range(i + 1, len(candidate_indices)):
                if rewards[i] == rewards[j]:
                    continue
                feature_i = candidate_feature(
                    validation_embeddings[query_index], train_embeddings[candidate_indices[i]]
                )
                feature_j = candidate_feature(
                    validation_embeddings[query_index], train_embeddings[candidate_indices[j]]
                )
                left.append(feature_i)
                right.append(feature_j)
                targets.append(1.0 if rewards[i] > rewards[j] else 0.0)
                query_ids.append(query_index)
    if not left:
        raise RuntimeError("No non-tied candidate pairs were found; rankings cannot be trained")
    return (
        np.stack(left),
        np.stack(right),
        np.asarray(targets, dtype=np.float32),
        np.asarray(query_ids, dtype=np.int64),
    )


def train_ranker(
    features_left: np.ndarray,
    features_right: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[RankNet, list[float]]:
    torch.manual_seed(seed)
    model = RankNet(features_left.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    left = torch.from_numpy(features_left)
    right = torch.from_numpy(features_right)
    target = torch.from_numpy(targets)
    generator = torch.Generator().manual_seed(seed)
    best_loss = math.inf
    stale_epochs = 0
    loss_history: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(left), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            x_left = left[indices].to(device)
            x_right = right[indices].to(device)
            y = target[indices].to(device)
            logits = model(x_left) - model(x_right)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(indices)
        epoch_loss /= len(left)
        loss_history.append(epoch_loss)
        if epoch == 1 or epoch % 10 == 0:
            print(f"    ranker epoch={epoch} loss={epoch_loss:.6f}", flush=True)
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"    ranker early stop at epoch {epoch}", flush=True)
                break
    model.eval()
    return model, loss_history


@torch.inference_mode()
def score_candidates(
    model: RankNet | None,
    query_embedding: np.ndarray,
    train_embeddings: np.ndarray,
    candidate_indices: list[list[int]],
    device: torch.device,
    fallback_candidate: int | None = None,
) -> tuple[int, list[float]]:
    if model is None:
        if fallback_candidate is None:
            raise ValueError("A fallback candidate is required when no ranker was trained")
        scores = [0.0] * len(candidate_indices)
        scores[fallback_candidate] = 1.0
        return fallback_candidate, scores
    features = np.stack(
        [candidate_feature(query_embedding, train_embeddings[indices]) for indices in candidate_indices]
    )
    scores = model(torch.from_numpy(features).to(device)).float().cpu().numpy().tolist()
    return int(np.argmax(scores)), scores


def load_feedback(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[int(record["candidate_id"])] = record
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_predictions(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    if args.method != "dearicl_4shot":
        raise ValueError("Only the DearICL 4-shot method is in scope")
    if args.shots != 4 or args.clusters != 4:
        raise ValueError("The configured plan requires 4 shots and 4 clusters")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for name in ("candidate_feedback.jsonl", "predictions.csv", "metrics.json", "selected_exemplars.json"):
            path = args.output_dir / name
            if path.exists():
                path.unlink()

    ds, question_key = load_local_dataset(args.data_root, args.dataset)
    train_rows = normalized_rows(ds, "train", question_key)
    validation_rows = normalized_rows(ds, "validation", question_key)
    test_rows = normalized_rows(ds, "test", question_key)
    validation_limit = args.limit_validation or args.validation_size
    validation_subset = stratified_indices(
        [row["answer"] for row in validation_rows], validation_limit, args.seed
    )
    validation_rows = [validation_rows[index] for index in validation_subset]
    if args.limit_test:
        test_rows = test_rows[: args.limit_test]
    if not 0 < args.shots <= min(len(test_rows), len(train_rows)):
        raise ValueError("Invalid shot configuration")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this experiment")
    torch.cuda.empty_cache()
    run_config = {
        "dataset": args.dataset,
        "model_path": args.model_path,
        "method": args.method,
        "shots": args.shots,
        "candidates": args.candidates,
        "clusters": args.clusters,
        "validation_size": validation_limit,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "train_size": len(train_rows),
        "validation_full_size": len(ds["validation"]),
        "test_size": len(test_rows),
    }
    write_json(args.output_dir / "run_config.json", run_config)

    offline_start = time.perf_counter()
    print(f"[{args.dataset}/{Path(args.model_path).name}] loading embedding model", flush=True)
    embedding_tokenizer, embedding_model = load_embedding_model(args.embedding_model, device)

    train_texts = [question_embedding_text(row) for row in train_rows]
    validation_texts = [question_embedding_text(row) for row in validation_rows]
    print(f"[{args.dataset}] encoding {len(train_texts)} train and {len(validation_texts)} validation questions", flush=True)
    train_embeddings = embed_texts(train_texts, embedding_tokenizer, embedding_model, device)
    validation_embeddings = embed_texts(validation_texts, embedding_tokenizer, embedding_model, device)
    candidates = build_candidates(
        train_embeddings, args.shots, args.candidates, args.clusters, args.seed
    )
    print(f"[{args.dataset}] generated {len(candidates)} candidate subsets", flush=True)

    print(f"[{args.dataset}/{Path(args.model_path).name}] loading generator", flush=True)
    generator_tokenizer, generator_model = load_generator(args.model_path, device)
    chat_model = "qwen" in Path(args.model_path).name.lower()

    feedback_path = args.output_dir / "candidate_feedback.jsonl"
    feedback_records = load_feedback(feedback_path)
    for candidate_id, indices in enumerate(candidates):
        existing = feedback_records.get(candidate_id)
        if existing and len(existing.get("rewards", [])) == len(validation_rows):
            continue
        print(
            f"[{args.dataset}/{Path(args.model_path).name}] validation feedback "
            f"candidate {candidate_id + 1}/{len(candidates)}",
            flush=True,
        )
        exemplars = [train_rows[index] for index in indices]
        rewards: list[int] = []
        answers: list[str | None] = []
        responses: list[str] = []
        for row_number, target in enumerate(validation_rows, start=1):
            prompt = build_prompt(target, exemplars, generator_tokenizer, chat_model)
            raw, prediction = generate_answer(
                generator_model,
                generator_tokenizer,
                prompt,
                args.max_new_tokens,
                args.max_input_tokens,
                "".join(target["labels"]),
            )
            rewards.append(int(prediction == target["answer"]))
            answers.append(prediction)
            responses.append(raw)
            if row_number % 25 == 0:
                print(
                    f"    validation {row_number}/{len(validation_rows)} "
                    f"running_acc={sum(rewards) / row_number:.3f}",
                    flush=True,
                )
        record = {
            "candidate_id": candidate_id,
            "indices": indices,
            "rewards": rewards,
            "predictions": answers,
            "responses": responses,
            "validation_accuracy": sum(rewards) / len(rewards),
        }
        append_jsonl(feedback_path, record)
        feedback_records[candidate_id] = record
        print(
            f"    candidate_accuracy={record['validation_accuracy']:.3f}",
            flush=True,
        )

    feedback = {
        candidate_id: [int(value) for value in feedback_records[candidate_id]["rewards"]]
        for candidate_id in range(len(candidates))
    }
    print(f"[{args.dataset}/{Path(args.model_path).name}] building ranking pairs", flush=True)
    try:
        features_left, features_right, targets, _ = build_ranking_pairs(
            validation_embeddings, train_embeddings, candidates, feedback
        )
    except RuntimeError:
        features_left = np.empty((0, train_embeddings.shape[1]), dtype=np.float32)
        features_right = features_left
        targets = np.empty((0,), dtype=np.float32)
    if len(features_left):
        print(f"    ranking pairs={len(features_left)}", flush=True)
        ranker, loss_history = train_ranker(
            features_left,
            features_right,
            targets,
            args.rank_epochs,
            args.rank_patience,
            args.rank_batch_size,
            args.seed,
            device,
        )
        torch.save(ranker.state_dict(), args.output_dir / "ranker.pt")
        fallback_candidate = None
        ranker_mode = "ranknet"
    else:
        candidate_accuracies = [
            sum(feedback[candidate_id]) / len(feedback[candidate_id])
            for candidate_id in range(len(candidates))
        ]
        fallback_candidate = int(np.argmax(candidate_accuracies))
        ranker = None
        loss_history = []
        ranker_mode = "global_best_validation_candidate"
        print(
            "    no non-tied ranking pairs; falling back to global validation-best candidate "
            f"{fallback_candidate}",
            flush=True,
        )
    offline_seconds = time.perf_counter() - offline_start
    print(f"[{args.dataset}/{Path(args.model_path).name}] offline_seconds={offline_seconds:.2f}", flush=True)

    prediction_path = args.output_dir / "predictions.csv"
    fieldnames = [
        "id",
        "question",
        "ground_truth",
        "prediction",
        "correct",
        "raw_output",
        "selected_candidate",
        "latency_ms",
    ]
    existing_rows = read_predictions(prediction_path)
    existing_by_id = {row["id"]: row for row in existing_rows}
    mode = "a" if prediction_path.exists() else "w"
    online_start = time.perf_counter()
    latencies_ms: list[float] = []
    selected_details: dict[str, Any] = {}
    completed_correct = sum(int(row["correct"]) for row in existing_rows)
    completed_count = len(existing_rows)
    with prediction_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        for row_number, target in enumerate(test_rows, start=1):
            if target["id"] in existing_by_id:
                existing = existing_by_id[target["id"]]
                latencies_ms.append(float(existing["latency_ms"]))
                candidate_id = int(existing["selected_candidate"])
                indices = candidates[candidate_id]
                selected_details[target["id"]] = {
                    "candidate_id": candidate_id,
                    "train_indices": indices,
                    "exemplar_ids": [train_rows[index]["id"] for index in indices],
                }
                continue
            item_start = time.perf_counter()
            query_embedding = embed_texts(
                [question_embedding_text(target)],
                embedding_tokenizer,
                embedding_model,
                device,
            )[0]
            candidate_id, candidate_scores = score_candidates(
                ranker,
                query_embedding,
                train_embeddings,
                candidates,
                device,
                fallback_candidate=fallback_candidate,
            )
            exemplars = [train_rows[index] for index in candidates[candidate_id]]
            prompt = build_prompt(target, exemplars, generator_tokenizer, chat_model)
            raw, prediction = generate_answer(
                generator_model,
                generator_tokenizer,
                prompt,
                args.max_new_tokens,
                args.max_input_tokens,
                "".join(target["labels"]),
            )
            latency_ms = (time.perf_counter() - item_start) * 1000.0
            latencies_ms.append(latency_ms)
            row = {
                "id": target["id"],
                "question": target["question"],
                "ground_truth": target["answer"],
                "prediction": prediction or "INVALID",
                "correct": "1" if prediction == target["answer"] else "0",
                "raw_output": raw,
                "selected_candidate": candidate_id,
                "latency_ms": f"{latency_ms:.3f}",
            }
            writer.writerow(row)
            handle.flush()
            completed_count += 1
            completed_correct += int(row["correct"])
            selected_details[target["id"]] = {
                "candidate_id": candidate_id,
                "candidate_scores": candidate_scores,
                "train_indices": candidates[candidate_id],
                "exemplar_ids": [train_rows[index]["id"] for index in candidates[candidate_id]],
            }
            if row_number == 1 or row_number % 50 == 0 or row_number == len(test_rows):
                print(
                    f"[{args.dataset}/{Path(args.model_path).name}] test "
                    f"{row_number}/{len(test_rows)} pred={prediction} gt={target['answer']} "
                    f"running_acc={completed_correct / max(1, completed_count):.3f}",
                    flush=True,
                )
    online_seconds = time.perf_counter() - online_start

    rows = read_predictions(prediction_path)
    y_true = [row["ground_truth"] for row in rows]
    y_pred = [row["prediction"] for row in rows]
    true_labels = sorted(set(y_true))
    metrics = {
        "dataset": args.dataset,
        "model_path": args.model_path,
        "method": args.method,
        "seed": args.seed,
        "test_samples": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=true_labels, average="weighted", zero_division=0)
        ),
        "run_time_s": offline_seconds + online_seconds,
        "offline_train_time_s": offline_seconds,
        "online_retrieval_time_s": online_seconds,
        "avg_latency_ms": float(np.mean(latencies_ms)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "invalid_count": sum(1 for value in y_pred if value == "INVALID"),
        "candidate_validation_accuracy": {
            str(candidate_id): feedback_records[candidate_id]["validation_accuracy"]
            for candidate_id in range(len(candidates))
        },
        "candidate_indices": {str(candidate_id): indices for candidate_id, indices in enumerate(candidates)},
        "ranker_mode": ranker_mode,
        "fallback_candidate": fallback_candidate,
        "rank_loss_history": loss_history,
        "config": run_config,
    }
    write_json(args.output_dir / "selected_exemplars.json", selected_details)
    write_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)

    del generator_model, generator_tokenizer, embedding_model, embedding_tokenizer
    if ranker is not None:
        del ranker
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
