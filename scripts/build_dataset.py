#!/usr/bin/env python3
"""
Dataset Builder for Summarization Model Training

Takes collected articles (from collect_articles.py) and creates a
training-ready dataset with auto-generated summaries.

Pipeline:
  1. Load collected articles JSON
  2. Clean & normalize text
  3. Generate summaries using a pre-trained model (BART)
  4. Split into train/validation/test JSONL
  5. Run quality checks

Output format per line (JSONL):
    {"article": "...", "summary": "..."}

Usage:
    python build_dataset.py --input data/collected_articles.json
    python build_dataset.py --input data/collected_articles.json --model sshleifer/distilbart-cnn-6-6
    python build_dataset.py --input data/collected_articles.json --use-titles
    python build_dataset.py --input data/collected_articles.json --split 0.8 0.1 0.1
    python build_dataset.py --input data/collected_articles.json --no-gpu
"""

import argparse
import json
import logging
import re
import statistics
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "data"


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_article(article):
    title = clean_text(article.get("title", ""))
    content = clean_text(article.get("content", ""))

    if content and title and not content.startswith(title):
        content = f"{title}. {content}"

    return {
        "title": title,
        "content": content,
        "source": article.get("source", ""),
        "url": article.get("url", ""),
        "date": article.get("date", ""),
    }


def filter_dataset(
    articles, min_article_words=50, max_article_words=2000, min_title_words=3
):
    filtered = []
    removed = {"too_short": 0, "too_long": 0, "no_title": 0, "empty": 0}

    for a in articles:
        content = a.get("content", "")
        title = a.get("title", "")

        if not content or len(content.split()) < min_article_words:
            removed["too_short"] += 1
            continue
        if len(content.split()) > max_article_words:
            removed["too_long"] += 1
            continue
        if not title or len(title.split()) < min_title_words:
            removed["no_title"] += 1
            continue
        if content == title:
            removed["empty"] += 1
            continue

        filtered.append(a)

    logger.info(f"Filtered: {len(articles)} -> {len(filtered)} articles")
    for reason, count in removed.items():
        if count > 0:
            logger.info(f"  Removed {count}: {reason}")

    return filtered


def generate_summaries_with_model(articles, model_name, use_gpu=True):
    import torch
    from transformers import pipeline

    device = 0 if use_gpu and torch.cuda.is_available() else -1
    logger.info(
        f"Loading model {model_name} (device={'cuda' if device >= 0 else 'cpu'})..."
    )

    summarizer = pipeline("summarization", model=model_name, device=device)

    results = []
    for i, article in enumerate(articles):
        content = article["content"]
        content_words = len(content.split())

        max_len = min(150, max(30, content_words // 3))
        min_len = max(20, max_len // 4)

        if content_words < 50:
            max_len = min(50, content_words - 5)
            min_len = max(10, max_len // 3)

        try:
            input_text = content[:3000]
            output = summarizer(
                input_text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True,
            )
            summary = output[0]["summary_text"]
        except Exception as e:
            logger.warning(f"  Failed to summarize article {i}: {e}")
            summary = article["title"]

        results.append(
            {
                "article": content,
                "summary": clean_text(summary),
            }
        )

        if (i + 1) % 10 == 0:
            logger.info(f"  Summarized {i + 1}/{len(articles)} articles")

    return results


def generate_summaries_from_titles(articles):
    results = []
    for article in articles:
        content = article["content"]
        summary = article["title"]
        results.append(
            {
                "article": content,
                "summary": clean_text(summary),
            }
        )
    return results


def split_dataset(data, train_ratio=0.8, val_ratio=0.1, seed=42):
    import random

    random.seed(seed)
    random.shuffle(data)

    train_end = int(len(data) * train_ratio)
    val_end = train_end + int(len(data) * val_ratio)

    return data[:train_end], data[train_end:val_end], data[val_end:]


def save_jsonl(data, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(data)} samples to {filepath}")


def quality_report(data, name):
    if not data:
        return
    art_lens = [len(r["article"].split()) for r in data]
    sum_lens = [len(r["summary"].split()) for r in data]
    ratios = [s / a if a > 0 else 0 for s, a in zip(sum_lens, art_lens)]

    logger.info(f"\n{name} ({len(data)} samples):")
    logger.info(
        f"  Article length: min={min(art_lens)}, max={max(art_lens)}, avg={statistics.mean(art_lens):.0f}"
    )
    logger.info(
        f"  Summary length: min={min(sum_lens)}, max={max(sum_lens)}, avg={statistics.mean(sum_lens):.0f}"
    )
    logger.info(f"  Compression ratio: avg={statistics.mean(ratios):.2f}")

    short_articles = [i for i, r in enumerate(data) if len(r["article"].split()) < 30]
    short_summaries = [i for i, r in enumerate(data) if len(r["summary"].split()) < 5]
    if short_articles:
        logger.info(f"  WARNING: {len(short_articles)} articles with < 30 words")
    if short_summaries:
        logger.info(f"  WARNING: {len(short_summaries)} summaries with < 5 words")


def main():
    parser = argparse.ArgumentParser(
        description="Build summarization dataset from collected articles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data/collected_articles.json
  %(prog)s --input data/collected_articles.json --model sshleifer/distilbart-cnn-6-6
  %(prog)s --input data/collected_articles.json --use-titles
  %(prog)s --input data/collected_articles.json --output-dir data/custom
  %(prog)s --input data/collected_articles.json --min-words 80 --max-words 1500
        """,
    )

    parser.add_argument(
        "--input", required=True, help="Path to collected articles JSON"
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--model",
        default="facebook/bart-large-cnn",
        help="Model for summary generation (default: facebook/bart-large-cnn)",
    )
    parser.add_argument(
        "--use-titles",
        action="store_true",
        help="Use titles as summaries instead of model generation",
    )
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU inference")
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test split ratios",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )
    parser.add_argument(
        "--min-words", type=int, default=50, help="Minimum article word count"
    )
    parser.add_argument(
        "--max-words", type=int, default=2000, help="Maximum article word count"
    )
    parser.add_argument(
        "--skip-quality-check", action="store_true", help="Skip quality report"
    )

    args = parser.parse_args()

    logger.info(f"Loading articles from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        raw_articles = json.load(f)
    logger.info(f"Loaded {len(raw_articles)} raw articles")

    articles = [normalize_article(a) for a in raw_articles]
    articles = filter_dataset(
        articles,
        min_article_words=args.min_words,
        max_article_words=args.max_words,
    )
    logger.info(f"After cleaning: {len(articles)} articles")

    if not articles:
        logger.error(
            "No articles left after filtering. Lower --min-words or check your data."
        )
        return

    if args.use_titles:
        logger.info("Using titles as summaries...")
        dataset = generate_summaries_from_titles(articles)
    else:
        logger.info(f"Generating summaries with {args.model}...")
        dataset = generate_summaries_with_model(
            articles, args.model, use_gpu=not args.no_gpu
        )

    train_ratio, val_ratio, test_ratio = args.split
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total

    train_data, val_data, test_data = split_dataset(
        dataset, train_ratio, val_ratio, seed=args.seed
    )

    output_dir = Path(args.output_dir)
    save_jsonl(train_data, output_dir / "train.jsonl")
    save_jsonl(val_data, output_dir / "validation.jsonl")
    save_jsonl(test_data, output_dir / "test.jsonl")

    if not args.skip_quality_check:
        quality_report(train_data, "TRAIN")
        quality_report(val_data, "VALIDATION")
        quality_report(test_data, "TEST")

    print(f"\n{'=' * 60}")
    print(f"Dataset built successfully!")
    print(f"  Train:      {len(train_data)} samples -> {output_dir / 'train.jsonl'}")
    print(f"  Validation: {len(val_data)} samples -> {output_dir / 'validation.jsonl'}")
    print(f"  Test:       {len(test_data)} samples -> {output_dir / 'test.jsonl'}")
    print(f"  Format:     JSONL with 'article' and 'summary' fields")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
