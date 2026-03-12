import argparse
import json
import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from config import get_processed_file_path, get_train_save_path

TRANSITION_SMOOTHING = 1e-6


def score_candidates_with_lm(
    model,
    tokenizer,
    prompt: str,
    candidates: list,
    device: str,
    sort: bool = True,
):


    model.eval()


    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]

    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

    scored_items = []

    for item in candidates:
        item_tokens = tokenizer.encode(item, add_special_tokens=False)
        if len(item_tokens) == 0:
            continue

        first_token_id = item_tokens[0]
        score = log_probs[first_token_id].item()
        scored_items.append((item, score))


    if sort:
        scored_items.sort(key=lambda x: x[1], reverse=True)
    return scored_items

def generate_cot(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens=256,
):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def build_transition_table(train_data):
    transitions = defaultdict(lambda: defaultdict(int))

    for sample in train_data:
        front = _ensure_list(sample.get("front"))
        results = _ensure_list(sample.get("result"))
        if not front or not results:
            continue

        if len(front) > 1:
            for i in range(len(front) - 1):
                transitions[front[i]][front[i + 1]] += 1

        transitions[front[-1]][results[0]] += 1

    return transitions


def _z_norm(scores):
    if not scores:
        return scores
    mean = sum(scores) / len(scores)
    var = sum((x - mean) ** 2 for x in scores) / len(scores)
    std = math.sqrt(var)
    if std < 1e-8:
        return [0.0 for _ in scores]
    return [(x - mean) / std for x in scores]


def compute_transition_scores(
    prev_item,
    candidates,
    transitions,
    smoothing=1e-6,
):
    if prev_item is None:
        return [0.0 for _ in candidates]
    counts = transitions.get(prev_item, {})
    scores = []
    for item in candidates:
        count = counts.get(item, 0)
        scores.append(math.log(count + smoothing))
    return scores


from tqdm import tqdm


def evaluate_variant2_rerank(
    model,
    tokenizer,
    test_data,
    device,
    k_list=(1, 5, 10, 20),
    save_path=None,
    transition_table=None,
    transition_alpha=1.0,
    transition_smoothing=1e-6,
):
    metric_sum = defaultdict(float)
    n = len(test_data)

    all_outputs = []

    for sample in tqdm(test_data, desc="Variant 2 (LM rerank)"):

        history = sample["history"]
        gt_item = sample["result"][0]
        candidates = sample["recommendations"]

        prompt = (
            f"User History: {history}\n"
            f"First explain your reasoning about user preferences, "
            f"then generate recommendations:\n"
            f"Recommendations:"
        )


        scored = score_candidates_with_lm(
            model, tokenizer, prompt, candidates, device, sort=False
        )
        lm_scores = [x[1] for x in scored]
        items = [x[0] for x in scored]

        if transition_table is not None and transition_alpha < 1.0:
            front = _ensure_list(sample.get("front"))
            prev_item = front[-1] if front else None
            trans_scores = compute_transition_scores(
                prev_item,
                items,
                transition_table,
                transition_smoothing,
            )

            lm_norm = _z_norm(lm_scores)
            trans_norm = _z_norm(trans_scores)
            combined = [
                transition_alpha * lm_norm[i]
                + (1.0 - transition_alpha) * trans_norm[i]
                for i in range(len(items))
            ]

            ranked_items = [
                x[0]
                for x in sorted(
                    zip(items, combined),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]
        else:
            ranked_items = [
                x[0]
                for x in sorted(
                    zip(items, lm_scores),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]


        rank = ranked_items.index(gt_item) + 1 if gt_item in ranked_items else None
        sample_metrics = compute_ranking_metrics(rank, k_list)

        for k, v in sample_metrics.items():
            metric_sum[k] += v


        all_outputs.append({
            "history": history,
            "ground_truth": gt_item,
            "ranked_items": ranked_items[:max(k_list)],
            "rank": rank,

            "metrics": sample_metrics,
        })


    return {k: v / n for k, v in metric_sum.items()}


def compute_ranking_metrics(rank: int, k_list=(1, 5, 10, 20)):

    metrics = {}

    for k in k_list:
        hit = rank is not None and rank <= k

        metrics[f"Recall@{k}"] = 1.0 if hit else 0.0
        metrics[f"Precision@{k}"] = (1.0 / k) if hit else 0.0

        if hit:
            metrics[f"NDCG@{k}"] = 1.0 / math.log2(rank + 1)
            metrics[f"MAP@{k}"] = 1.0 / rank
        else:
            metrics[f"NDCG@{k}"] = 0.0
            metrics[f"MAP@{k}"] = 0.0

    return metrics


from transformers import AutoTokenizer, AutoModelForCausalLM

def metric_sort_key(metric_name: str):
    name, k = metric_name.split("@")
    return (name, int(k))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="ml100k",
        choices=["ml100k", "ml1m", "electronics", "movies"],
    )
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Train file for transition prior (auto from config if transition_alpha < 1.0)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument(
        "--transition_alpha",
        type=float,
        default=0.5,
        help="LM weight in [0,1]. Default is 0.5.",
    )
    args = parser.parse_args()

    if args.model_dir is None:
        args.model_dir = str(get_train_save_path(args.dataset) / "final")
    if args.test_file is None:
        args.test_file = str(get_processed_file_path(args.dataset, "test"))
    if args.train_file is None and args.transition_alpha < 1.0:
        args.train_file = str(get_processed_file_path(args.dataset, "train"))

    print(f"Model dir: {args.model_dir}")
    print(f"Test file: {args.test_file}")
    if args.train_file is not None:
        print(f"Train file: {args.train_file}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(args.device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.test_file) as f:
        test_data = json.load(f)

    transition_table = None
    if args.transition_alpha < 1.0:
        if args.train_file is None:
            raise SystemExit(
                "transition_alpha < 1.0 requires --train_file"
            )
        with open(args.train_file) as f:
            train_data = json.load(f)
        transition_table = build_transition_table(train_data)

    metrics = evaluate_variant2_rerank(
        model,
        tokenizer,
        test_data,
        args.device,
        (1, 5, 10, 20),
        args.save_path,
        transition_table=transition_table,
        transition_alpha=args.transition_alpha,
        transition_smoothing=TRANSITION_SMOOTHING,
    )

    print("\n===== Results =====")
    for key in sorted(metrics.keys(), key=metric_sort_key):
        print(f"{key}: {metrics[key]:.4f}")
