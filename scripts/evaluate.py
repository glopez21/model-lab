"""
Evaluate a summarization model using ROUGE metrics.

Usage:
    python evaluate.py --model results/cyber-bart --data data/test.jsonl
    python evaluate.py --model facebook/bart-large-cnn --data data/test.jsonl --split compare
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from evaluate import load as load_metric
from transformers import pipeline


def load_data(data_path: str) -> list[dict]:
    rows = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(args):
    rows = load_data(args.data)
    summarizer = pipeline("summarization", model=args.model, device_map="auto")

    rouge = load_metric("rouge")

    predictions = []
    references = []

    for i, row in enumerate(rows):
        article = row["article"][: args.max_input_len]
        try:
            result = summarizer(
                article,
                max_length=args.max_target_len,
                min_length=args.min_target_len,
                do_sample=False,
            )
            pred = result[0]["summary_text"]
        except Exception as e:
            print(f"Error on row {i}: {e}")
            pred = ""

        predictions.append(pred)
        references.append(row["summary"])

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(rows)}")

    scores = rouge.compute(predictions=predictions, references=references)

    print(f"\n=== Evaluation: {args.model} ===")
    print(f"Samples: {len(rows)}")
    for key in ["rouge1", "rouge2", "rougeL"]:
        print(f"  {key}: {scores[key].mid.fmeasure:.4f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "model": args.model,
            "samples": len(rows),
            "rouge1": scores["rouge1"].mid.fmeasure,
            "rouge2": scores["rouge2"].mid.fmeasure,
            "rougeL": scores["rougeL"].mid.fmeasure,
        }
        # Save individual predictions
        results["predictions"] = [
            {"article_id": i, "reference": r, "prediction": p}
            for i, (r, p) in enumerate(zip(references, predictions))
        ]
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a summarization model")
    parser.add_argument(
        "--model", required=True, help="Model path or HuggingFace model name"
    )
    parser.add_argument("--data", required=True, help="Test data (JSONL)")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument(
        "--max-input-len", type=int, default=4000, help="Max input characters"
    )
    parser.add_argument(
        "--max-target-len", type=int, default=150, help="Max summary tokens"
    )
    parser.add_argument(
        "--min-target-len", type=int, default=30, help="Min summary tokens"
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
