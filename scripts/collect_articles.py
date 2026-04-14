#!/usr/bin/env python3
"""
Article Collector for Model Training

Scrapes cybersecurity news articles from multiple sources and exports
them as clean JSON for dataset building.

Output format (per article):
    {
        "title": "...",
        "content": "...",
        "source": "...",
        "url": "...",
        "date": "..."
    }

Can also import articles from the existing articles.db or scraped JSON files.

Usage:
    python collect_articles.py                                     # scrape all sources
    python collect_articles.py --sources thehackernews bleepingcomputer
    python collect_articles.py --from-db ../../news_articles/articles.db
    python collect_articles.py --from-json ../../news_articles/scrapers/*.json
    python collect_articles.py --sources thehackernews --max 30
    python collect_articles.py --output my_data.json
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "data" / "collected_articles.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

SOURCES = {
    "thehackernews": {
        "name": "The Hacker News",
        "base_url": "https://thehackernews.com",
        "rss_url": "https://feeds.feedburner.com/TheHackersNews",
        "link_pattern": "/2026/",
    },
    "bleepingcomputer": {
        "name": "BleepingComputer",
        "base_url": "https://www.bleepingcomputer.com",
        "rss_url": "https://www.bleepingcomputer.com/feed/",
        "link_pattern": "/news/",
    },
    "darkreading": {
        "name": "Dark Reading",
        "base_url": "https://www.darkreading.com",
        "rss_url": "https://www.darkreading.com/rss.xml",
        "link_pattern": "/",
    },
    "threatpost": {
        "name": "Threatpost",
        "base_url": "https://threatpost.com",
        "rss_url": "https://threatpost.com/feed/",
        "link_pattern": "/",
    },
    "networkworld": {
        "name": "NetworkWorld",
        "base_url": "https://www.networkworld.com",
        "rss_url": "https://www.networkworld.com/feed/",
        "link_pattern": "/article/",
    },
    "theregister": {
        "name": "The Register",
        "base_url": "https://www.theregister.com",
        "rss_url": "https://www.theregister.com/headlines.rss",
        "link_pattern": "/2026/",
    },
    "cybernews": {
        "name": "CyberNews",
        "base_url": "https://cybernews.com",
        "rss_url": "https://cybernews.com/feed/",
        "link_pattern": "/news/",
    },
    "gbhackers": {
        "name": "GBHackers",
        "base_url": "https://gbhackers.com",
        "rss_url": "https://gbhackers.com/feed/",
        "link_pattern": "/news/",
    },
}


def article_hash(url, title):
    return hashlib.sha256(f"{url}{title}".encode()).hexdigest()


def get_article_links(source_info, max_articles):
    try:
        response = requests.get(source_info["base_url"], headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "lxml")

        links = []
        seen = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href:
                continue
            if not href.startswith("http"):
                href = source_info["base_url"].rstrip("/") + href
            if source_info["link_pattern"] in href and href not in seen:
                domain = urlparse(href).netloc
                source_domain = urlparse(source_info["base_url"]).netloc
                if domain == source_domain or domain == f"www.{source_domain}":
                    seen.add(href)
                    links.append(href)
            if len(links) >= max_articles:
                break

        return links
    except Exception as e:
        logger.error(f"Error fetching links from {source_info['name']}: {e}")
        return []


def parse_rss_feed(source_key, source_info, max_articles):
    feed_url = source_info.get("rss_url")
    if not feed_url:
        return []
    try:
        response = requests.get(feed_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "xml")

        articles = []
        for item in soup.find_all("item")[:max_articles]:
            title_tag = item.find("title")
            link_tag = item.find("link")
            pub_date = item.find("pubDate")
            content_tag = item.find("content:encoded") or item.find("description")
            author_tag = item.find("author") or item.find("dc:creator")

            url = link_tag.get_text(strip=True) if link_tag else ""
            title = title_tag.get_text(strip=True) if title_tag else "No title"
            content = ""
            if content_tag:
                content_soup = BeautifulSoup(content_tag.get_text(), "lxml")
                paragraphs = content_soup.find_all("p")
                if paragraphs:
                    content = " ".join(
                        p.get_text(strip=True)
                        for p in paragraphs
                        if p.get_text(strip=True)
                    )
                if not content.strip():
                    content = content_tag.get_text(strip=True)

            date = ""
            if pub_date:
                try:
                    dt = parsedate_to_datetime(pub_date.get_text())
                    date = dt.strftime("%B %d, %Y")
                except Exception:
                    date = pub_date.get_text(strip=True)[:10]

            if not content.strip():
                content = title

            articles.append(
                {
                    "title": title,
                    "content": content,
                    "source": source_info["name"],
                    "url": url,
                    "date": date,
                }
            )
        return articles
    except Exception as e:
        logger.error(f"RSS error for {source_info['name']}: {e}")
        return []


PARSERS = {}


def parse_thehackernews(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "lxml")

        title = "No title"
        content = ""
        date = ""

        og_title = soup.find("meta", property="og:title")
        if og_title:
            title = og_title.get("content", "No title")

        date_meta = soup.find("meta", itemprop="datePublished")
        if date_meta:
            date_str = date_meta.get("content", "")
            try:
                date = datetime.fromisoformat(date_str.split("T")[0]).strftime(
                    "%B %d, %Y"
                )
            except Exception:
                date = date_str[:10] if date_str else ""

        post_body = soup.find("div", class_="post-body")
        if post_body:
            paragraphs = post_body.find_all("p")
            content = " ".join(
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            )

        return {"title": title, "content": content or title, "date": date}
    except Exception as e:
        logger.error(f"Error parsing {url}: {e}")
        return None


def parse_bleepingcomputer(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "lxml")

        title_tag = soup.find("h2", class_="article__title") or soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "No title"

        date_tag = soup.find("time")
        date = ""
        if date_tag and date_tag.get("datetime"):
            try:
                date = datetime.fromisoformat(date_tag["datetime"][:10]).strftime(
                    "%B %d, %Y"
                )
            except Exception:
                date = date_tag["datetime"][:10]

        content_tag = soup.find("div", class_="article__content")
        content = ""
        if content_tag:
            paragraphs = content_tag.find_all("p")
            content = " ".join(
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            )

        return {"title": title, "content": content or title, "date": date}
    except Exception as e:
        logger.error(f"Error parsing {url}: {e}")
        return None


def parse_threatpost(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "lxml")

        title_tag = soup.find("h1", class_="article-title") or soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "No title"

        date_tag = soup.find("time")
        date = ""
        if date_tag and date_tag.get("datetime"):
            try:
                date = datetime.fromisoformat(date_tag["datetime"][:10]).strftime(
                    "%B %d, %Y"
                )
            except Exception:
                date = date_tag["datetime"][:10]

        content_tag = soup.find("div", class_="article-body")
        content = ""
        if content_tag:
            paragraphs = content_tag.find_all("p")
            content = " ".join(
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            )

        return {"title": title, "content": content or title, "date": date}
    except Exception as e:
        logger.error(f"Error parsing {url}: {e}")
        return None


def parse_darkreading(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "lxml")

        title_tag = soup.find("h1", class_="article-title") or soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "No title"

        date_tag = soup.find("time")
        date = ""
        if date_tag and date_tag.get("datetime"):
            try:
                date = datetime.fromisoformat(date_tag["datetime"][:10]).strftime(
                    "%B %d, %Y"
                )
            except Exception:
                date = date_tag["datetime"][:10]

        content_tag = soup.find("div", class_="article-body")
        content = ""
        if content_tag:
            paragraphs = content_tag.find_all("p")
            content = " ".join(
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            )

        return {"title": title, "content": content or title, "date": date}
    except Exception as e:
        logger.error(f"Error parsing {url}: {e}")
        return None


def parse_generic_article(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "lxml")

        title = "No title"
        og_title = soup.find("meta", property="og:title")
        if og_title:
            title = og_title.get("content", "No title")
        else:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)

        date = ""
        for selector in [
            ("meta", {"property": "article:published_time"}),
            ("meta", {"itemprop": "datePublished"}),
        ]:
            tag = soup.find(selector[0], selector[1])
            if tag:
                date_str = tag.get("content", "")
                try:
                    date = datetime.fromisoformat(date_str.split("T")[0]).strftime(
                        "%B %d, %Y"
                    )
                except Exception:
                    date = date_str[:10]
                break

        content = ""
        for container in [
            soup.find("article"),
            soup.find("div", class_="article-body"),
            soup.find("div", class_="post-body"),
            soup.find("div", class_="entry-content"),
            soup.find("div", class_="article__content"),
        ]:
            if container:
                paragraphs = container.find_all("p")
                content = " ".join(
                    p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
                )
                if content:
                    break

        if not content:
            paragraphs = soup.find_all("p")
            content = " ".join(
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            )

        return {"title": title, "content": content or title, "date": date}
    except Exception as e:
        logger.error(f"Error parsing {url}: {e}")
        return None


PARSERS = {
    "thehackernews": parse_thehackernews,
    "bleepingcomputer": parse_bleepingcomputer,
    "threatpost": parse_threatpost,
    "darkreading": parse_darkreading,
}


def scrape_source(source_key, max_articles):
    source_info = SOURCES[source_key]
    logger.info(f"Scraping {source_info['name']}...")

    links = get_article_links(source_info, max_articles)
    logger.info(f"  Found {len(links)} links from homepage")

    articles = []
    errors = 0

    if links and len(links) > 3:
        parser = PARSERS.get(source_key, parse_generic_article)
        iterator = tqdm(links, desc=source_info["name"]) if TQDM_AVAILABLE else links

        for url in iterator:
            result = parser(url)
            if result:
                articles.append(
                    {
                        "title": result["title"],
                        "content": result["content"],
                        "source": source_info["name"],
                        "url": url,
                        "date": result.get("date", ""),
                    }
                )
            else:
                errors += 1
            time.sleep(0.5)

        success_rate = len(articles) / len(links) * 100 if links else 0
        if success_rate < 50:
            logger.warning(
                f"Low success rate ({success_rate:.0f}%), trying RSS for {source_info['name']}"
            )
            articles = parse_rss_feed(source_key, source_info, max_articles)
    else:
        logger.info(f"  Not enough links from homepage, trying RSS...")
        articles = parse_rss_feed(source_key, source_info, max_articles)

    logger.info(
        f"  Got {len(articles)} articles from {source_info['name']} ({errors} errors)"
    )
    return articles


def deduplicate(articles):
    seen = set()
    unique = []
    for a in articles:
        h = article_hash(a.get("url", ""), a.get("title", ""))
        if h not in seen:
            seen.add(h)
            unique.append(a)
    return unique


def filter_articles(articles, min_words=30):
    filtered = []
    for a in articles:
        content = a.get("content", "")
        word_count = len(content.split())
        if word_count < min_words:
            continue
        if a.get("title", "No title") == "No title" and word_count < 50:
            continue
        filtered.append(a)
    return filtered


def import_from_db(db_path):
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT source, title, content, url, date FROM articles")
    rows = cursor.fetchall()
    conn.close()

    articles = []
    for row in rows:
        articles.append(
            {
                "source": row[0] or "",
                "title": row[1] or "",
                "content": row[2] or "",
                "url": row[3] or "",
                "date": row[4] or "",
            }
        )

    logger.info(f"Imported {len(articles)} articles from database")
    return articles


def import_from_json_files(file_paths):
    articles = []
    for fp in file_paths:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    articles.append(
                        {
                            "source": item.get("source", ""),
                            "title": item.get("title", ""),
                            "content": item.get("content", ""),
                            "url": item.get("url", ""),
                            "date": item.get("date", ""),
                        }
                    )
            elif isinstance(data, dict):
                articles.append(data)
            logger.info(
                f"Loaded {len(data) if isinstance(data, list) else 1} articles from {fp}"
            )
        except Exception as e:
            logger.error(f"Error loading {fp}: {e}")
    return articles


def import_from_csv_files(file_paths):
    articles = []
    for fp in file_paths:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    articles.append(
                        {
                            "source": row.get("source", ""),
                            "title": row.get("title", ""),
                            "content": row.get("content", ""),
                            "url": row.get("url", ""),
                            "date": row.get("date", ""),
                        }
                    )
            logger.info(f"Loaded articles from {fp}")
        except Exception as e:
            logger.error(f"Error loading {fp}: {e}")
    return articles


def save_output(articles, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(articles)} articles to {output_path}")

    sources_summary = {}
    for a in articles:
        src = a.get("source", "Unknown")
        sources_summary[src] = sources_summary.get(src, 0) + 1

    logger.info("Sources breakdown:")
    for src, count in sorted(sources_summary.items()):
        logger.info(f"  {src}: {count}")

    word_counts = [len(a["content"].split()) for a in articles]
    if word_counts:
        logger.info(
            f"Content stats: min={min(word_counts)}, max={max(word_counts)}, "
            f"avg={sum(word_counts) // len(word_counts)} words"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Collect news articles for model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                             # Scrape all sources
  %(prog)s --sources thehackernews darkreading          # Specific sources
  %(prog)s --from-db ../../news_articles/articles.db    # Import from existing DB
  %(prog)s --from-json ../../news_articles/scrapers/thehackernews_articles.json
  %(prog)s --sources thehackernews --max 20             # Limit per source
  %(prog)s --output data/my_collection.json             # Custom output path

Available sources: """
        + ", ".join(SOURCES.keys()),
    )

    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(SOURCES.keys()),
        default=list(SOURCES.keys()),
        help="Sources to scrape (default: all)",
    )
    parser.add_argument(
        "--max", type=int, default=50, help="Max articles per source (default: 50)"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=30,
        help="Minimum content word count (default: 30)",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT), help="Output JSON file path"
    )
    parser.add_argument(
        "--from-db", help="Import articles from SQLite database instead of scraping"
    )
    parser.add_argument(
        "--from-json", nargs="+", help="Import articles from JSON file(s)"
    )
    parser.add_argument(
        "--from-csv", nargs="+", help="Import articles from CSV file(s)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge with existing output file instead of overwriting",
    )

    args = parser.parse_args()

    all_articles = []

    if args.from_db:
        all_articles.extend(import_from_db(args.from_db))

    if args.from_json:
        all_articles.extend(import_from_json_files(args.from_json))

    if args.from_csv:
        all_articles.extend(import_from_csv_files(args.from_csv))

    if not (args.from_db or args.from_json or args.from_csv):
        logger.info("Scraping articles from web...")
        for source_key in args.sources:
            articles = scrape_source(source_key, args.max)
            all_articles.extend(articles)

    all_articles = deduplicate(all_articles)
    all_articles = filter_articles(all_articles, min_words=args.min_words)

    if args.merge and os.path.exists(args.output):
        existing = import_from_json_files([args.output])
        all_articles.extend(existing)
        all_articles = deduplicate(all_articles)

    save_output(all_articles, args.output)

    print(f"\n{'=' * 60}")
    print(f"Collection complete: {len(all_articles)} articles saved")
    print(f"Output: {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
