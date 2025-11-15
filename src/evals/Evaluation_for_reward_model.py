# -*- coding: utf-8 -*-
"""
evaluate_reward_model_v2.py

Full evaluation pipeline for Reward Model calibration and performance.

Evaluates:
- Pairwise accuracy on preference pairs
- Spearman correlation with human/heuristic confidence
- Calibration (ECE)
- Penalty response on high-risk overconfidence
- Risk-tier breakdown and summary plots

Author: DSA4213 Group 18 (Revised)
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==========================================================
# 1️⃣ Utility Functions
# ==========================================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_input(question, answer):
    """Format question-answer pair for reward model."""
    return f"Question: {question}\nAnswer: {answer}"


# ==========================================================
# 2️⃣ Reward Scoring
# ==========================================================
@torch.no_grad()
def compute_reward_batch(model, tokenizer, qa_list, device="cuda", batch_size=8):
    """Compute reward model scores efficiently in batches."""
    model.eval()
    all_scores = []

    for i in tqdm(range(0, len(qa_list), batch_size), desc="Computing rewards"):
        batch = qa_list[i : i + batch_size]
        texts = [prepare_input(q, a) for q, a in batch]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)
        outputs = model(**inputs)
        scores = outputs.logits.squeeze(-1).detach().cpu().numpy().tolist()
        all_scores.extend(scores)

    return all_scores
    
# ==========================================================
# 3️⃣ Evaluation Metrics
# ==========================================================
def evaluate_pairwise_accuracy(model, tokenizer, pairs, device="cuda"):
    """Compute accuracy on preference pairs."""
    correct = 0
    for p in tqdm(pairs, desc="Evaluating preference pairs"):
        q = p["question_text"]
        a_pref = p["preferred_answer"]["answer_text"]
        a_rej = p["rejected_answer"]["answer_text"]

        r_pref = compute_reward(model, tokenizer, q, a_pref, device)
        r_rej = compute_reward(model, tokenizer, q, a_rej, device)
        if r_pref > r_rej:
            correct += 1
    return correct / len(pairs)


def evaluate_correlation(model, tokenizer, qna_data, device="cuda"):
    """Compute Spearman correlation with human scores."""
    rm_scores, human_scores = [], []
    for item in tqdm(qna_data, desc="Evaluating correlation"):
        q = item["question"]
        for ans in item["answers"]:
            if "human_score" not in ans:  # skip if no human score
                continue
            rm_score = compute_reward(model, tokenizer, q, ans["answer_text"], device)
            rm_scores.append(rm_score)
            human_scores.append(ans["human_score"])
    if len(rm_scores) < 5:
        return None
    rho, _ = spearmanr(rm_scores, human_scores)
    return rho


def compute_calibration(model, tokenizer, qna_data, device="cuda"):
    """Compute calibration curve and Expected Calibration Error (ECE)."""
    preds, confs = [], []
    for item in tqdm(qna_data, desc="Evaluating calibration"):
        q = item["question"]
        for ans in item["answers"]:
            if "human_score" not in ans:
                continue
            score = compute_reward(model, tokenizer, q, ans["answer_text"], device)
            conf = ans.get("confidence_score", 0.5)
            # Simplify: treat human_score >= 3 (Good/Fair) as correct
            label = 1 if ans["human_score"] >= 3 else 0
            preds.append(torch.sigmoid(torch.tensor(score)).item())
            confs.append((conf, label))

    # Compute Expected Calibration Error (ECE)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = [(p, l) for (p, (c, l)) in zip(preds, confs) if bin_lower < c <= bin_upper]
        if len(in_bin) == 0:
            continue
        acc_bin = np.mean([l for _, (_, l) in zip(preds, confs) if bin_lower < _ <= bin_upper])
        conf_bin = np.mean([c for (c, l) in [pair[1] for pair in in_bin]])
        ece += len(in_bin) / len(preds) * abs(acc_bin - conf_bin)
    return round(ece, 4)


def evaluate_penalty_effect(model, tokenizer, qna_data, device="cuda"):
    """Check how reward responds to overconfident answers in high-risk contexts."""
    deltas = []
    for item in tqdm(qna_data, desc="Evaluating penalty response"):
        if item.get("risk_level", "Low") != "High":
            continue
        q = item["question"]
        answers = item["answers"]
        if len(answers) < 2:
            continue
        sorted_ans = sorted(answers, key=lambda a: a.get("confidence_score", 0.5), reverse=True)
        high_conf = sorted_ans[0]
        low_conf = sorted_ans[-1]
        r_high = compute_reward(model, tokenizer, q, high_conf["answer_text"], device)
        r_low = compute_reward(model, tokenizer, q, low_conf["answer_text"], device)
        deltas.append(r_high - r_low)
    if len(deltas) == 0:
        return None
    return round(np.mean(deltas), 4)


# ==========================================================
# 4️⃣ Risk-Tier Diagnostics
# ==========================================================
def evaluate_by_risk_tier(model, tokenizer, qna_data, device="cuda"):
    results = {}
    risk_levels = ["High", "Medium", "Low"]
    for risk in risk_levels:
        subset = [x for x in qna_data if x.get("risk_level", "Low") == risk]
        if not subset:
            continue
        rho = evaluate_correlation(model, tokenizer, subset, device)
        results[risk] = {"correlation": rho, "count": len(subset)}
    return results


# ==========================================================
# 5️⃣ Main Evaluation Orchestration
# ==========================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Reward Model")
    parser.add_argument("--model_path", required=True, help="Path to trained reward model")
    parser.add_argument("--pairs", required=True, help="Path to preference pairs JSON")
    parser.add_argument("--test", required=True, help="Path to test Q&A JSON with confidence")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="outputs/evaluation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {args.device}")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(args.device)

    print("Loading data...")
    pairs = load_json(args.pairs)
    qna_data = load_json(args.test)

    # ------------- Run Evaluations ------------------
    print("\nRunning Reward Model Evaluation...")
    metrics = {}

    metrics["pairwise_accuracy"] = evaluate_pairwise_accuracy(model, tokenizer, pairs, args.device)
    metrics["spearman_correlation"] = evaluate_correlation(model, tokenizer, qna_data, args.device)
    metrics["ECE"] = compute_calibration(model, tokenizer, qna_data, args.device)
    metrics["mean_penalty_delta"] = evaluate_penalty_effect(model, tokenizer, qna_data, args.device)
    metrics["risk_tier_summary"] = evaluate_by_risk_tier(model, tokenizer, qna_data, args.device)

    # Save metrics
    out_path = os.path.join(args.output_dir, "reward_model_eval_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Saved metrics to {out_path}")

    # Print summary
    print("\nReward Model Evaluation Summary")
    print("=" * 40)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:25s}: {v:.4f}")
        else:
            print(f"{k:25s}: {v}")

    # Plot calibration (optional)
    try:
        conf_scores = [a["confidence_score"] for q in qna_data for a in q["answers"] if "confidence_score" in a]
        rewards = [compute_reward(model, tokenizer, q["question"], a["answer_text"], args.device)
                   for q in qna_data for a in q["answers"] if "confidence_score" in a]
        plt.scatter(conf_scores, rewards, alpha=0.4)
        plt.xlabel("Confidence Score")
        plt.ylabel("Reward Model Output")
        plt.title("Calibration Scatter: Confidence vs Reward")
        plt.savefig(os.path.join(args.output_dir, "calibration_scatter.png"))
        print("Saved calibration plot.")
    except Exception as e:
        print(f"Could not plot calibration: {e}")


if __name__ == "__main__":
    main()


