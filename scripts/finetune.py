"""
Fine-tune a summarization model on cybersecurity news data.

Usage:
    python finetune.py --base-model facebook/bart-large-cnn --data data/train.jsonl --output results/cyber-bart
    python finetune.py --base-model sshleifer/distilbart-cnn-6-6 --data data/train.jsonl --output results/cyber-distilbart
    python finetune.py --base-model google/pegasus-cnn_dailymail --data data/train.jsonl --output results/cyber-pegasus
"""

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset
from evaluate import load as load_metric
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)


def load_data(data_path: str) -> Dataset:
    """Load JSONL training data. Each line: {"article": "...", "summary": "..."}"""
    rows = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def preprocess(batch, tokenizer, max_input_len, max_target_len):
    inputs = tokenizer(
        batch["article"],
        max_length=max_input_len,
        truncation=True,
        padding="max_length",
    )
    targets = tokenizer(
        batch["summary"],
        max_length=max_target_len,
        truncation=True,
        padding="max_length",
    )
    inputs["labels"] = targets["input_ids"]
    return inputs


def finetune(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    train_dataset = load_data(args.data)

    if args.eval_data and os.path.exists(args.eval_data):
        eval_dataset = load_data(args.eval_data)
    else:
        split = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    train_dataset = train_dataset.map(
        lambda b: preprocess(b, tokenizer, args.max_input_len, args.max_target_len),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda b: preprocess(b, tokenizer, args.max_input_len, args.max_target_len),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        fp16=args.fp16,
        logging_dir=os.path.join(args.output, "logs"),
        logging_steps=50,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.grad_accum,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nModel saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a summarization model")
    parser.add_argument(
        "--base-model",
        default="facebook/bart-large-cnn",
        help="Base model to fine-tune",
    )
    parser.add_argument("--data", required=True, help="Path to training data (JSONL)")
    parser.add_argument("--eval-data", default=None, help="Path to eval data (JSONL)")
    parser.add_argument(
        "--output", default="results/cyber-summarizer", help="Output directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=3e-5, help="Learning rate"
    )
    parser.add_argument(
        "--max-input-len", type=int, default=1024, help="Max input token length"
    )
    parser.add_argument(
        "--max-target-len", type=int, default=150, help="Max target token length"
    )
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps")
    parser.add_argument(
        "--grad-accum", type=int, default=2, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Use mixed precision training"
    )
    args = parser.parse_args()
    finetune(args)


if __name__ == "__main__":
    main()
