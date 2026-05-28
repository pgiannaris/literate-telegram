#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote_plus, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_QUERIES = [
    "vaccine hesitancy",
    "health hesitancy",
    "vaccine skepticism",
    "vaccine safety concerns",
]

DEFAULT_SUBREDDITS = [
    "vaccines",
    "medicine",
    "Coronavirus",
    "science",
    "AskDocs",
]


@dataclasses.dataclass
class Record:
    id: str
    source: str
    source_detail: str
    query: str
    url: str
    title: str
    text: str
    author: str
    published_at: str
    collected_at: str
    stance: str = ""
    sentiment: str = ""
    confidence: float = 0.0
    reason: str = ""


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def load_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "health-hesitancy-sentiment-scraper/1.0 "
                "(research script; contact: local-user)"
            )
        }
    )
    return session


def html_fragment_to_text(value: str) -> str:
    if not value:
        return ""
    soup = BeautifulSoup(value, "html.parser")
    return normalize_space(" ".join(soup.stripped_strings))


def extract_main_text(html: str, max_chars: int) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "form", "iframe"]):
        tag.decompose()

    candidate = soup.find("article")
    if candidate is None:
        candidate = soup.body or soup

    text = normalize_space(" ".join(candidate.stripped_strings))
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def fetch_webpage_text(
    session: requests.Session,
    url: str,
    request_delay: float,
    max_chars: int = 7000,
) -> str:
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        logging.warning("Failed to fetch page %s: %s", url, exc)
        return ""
    finally:
        if request_delay > 0:
            time.sleep(request_delay)

    content_type = response.headers.get("content-type", "")
    if "html" not in content_type:
        return ""
    return extract_main_text(response.text, max_chars=max_chars)


def reddit_timestamp_to_iso(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return ""
    timestamp = dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)
    return timestamp.isoformat()


def collect_reddit(
    session: requests.Session,
    subreddits: Sequence[str],
    queries: Sequence[str],
    max_per_query: int,
    request_delay: float,
) -> list[Record]:
    records: list[Record] = []
    collected_at = now_iso()

    for subreddit in subreddits:
        for query in queries:
            fetched = 0
            after = ""
            while fetched < max_per_query:
                limit = min(100, max_per_query - fetched)
                params = {
                    "q": query,
                    "restrict_sr": "on",
                    "sort": "new",
                    "t": "year",
                    "limit": str(limit),
                }
                if after:
                    params["after"] = after

                url = f"https://www.reddit.com/r/{subreddit}/search.json"
                try:
                    response = session.get(url, params=params, timeout=30)
                    if response.status_code == 404:
                        logging.warning("Subreddit not found: r/%s", subreddit)
                        break
                    response.raise_for_status()
                    payload = response.json()
                except requests.RequestException as exc:
                    logging.warning(
                        "Reddit request failed for r/%s query='%s': %s",
                        subreddit,
                        query,
                        exc,
                    )
                    break
                except json.JSONDecodeError as exc:
                    logging.warning(
                        "Invalid Reddit JSON for r/%s query='%s': %s",
                        subreddit,
                        query,
                        exc,
                    )
                    break
                finally:
                    if request_delay > 0:
                        time.sleep(request_delay)

                children = payload.get("data", {}).get("children", [])
                if not children:
                    break

                for item in children:
                    post = item.get("data", {})
                    title = normalize_space(post.get("title", ""))
                    body = normalize_space(post.get("selftext", ""))
                    text = normalize_space(f"{title}\n{body}")
                    if not text:
                        continue

                    post_id = str(post.get("id", "")) or stable_hash(text[:120])
                    permalink = post.get("permalink", "")
                    final_url = (
                        f"https://www.reddit.com{permalink}"
                        if permalink
                        else post.get("url", "")
                    )
                    author = str(post.get("author", ""))
                    published_at = reddit_timestamp_to_iso(post.get("created_utc"))
                    source_detail = f"r/{subreddit}"

                    records.append(
                        Record(
                            id=stable_hash(f"reddit|{subreddit}|{post_id}"),
                            source="reddit",
                            source_detail=source_detail,
                            query=query,
                            url=final_url,
                            title=title,
                            text=text,
                            author=author,
                            published_at=published_at,
                            collected_at=collected_at,
                        )
                    )
                    fetched += 1
                    if fetched >= max_per_query:
                        break

                after = payload.get("data", {}).get("after", "")
                if not after:
                    break

    return records


def google_news_feed_url(query: str, days_back: int) -> str:
    q = quote_plus(f"{query} when:{days_back}d")
    return (
        f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    )


def collect_google_news(
    session: requests.Session,
    queries: Sequence[str],
    days_back: int,
    max_per_query: int,
    request_delay: float,
) -> list[Record]:
    records: list[Record] = []
    collected_at = now_iso()

    for query in queries:
        feed_url = google_news_feed_url(query, days_back=days_back)
        parsed = feedparser.parse(feed_url)
        entries = parsed.entries[:max_per_query]

        for entry in entries:
            title = normalize_space(getattr(entry, "title", ""))
            link = normalize_space(getattr(entry, "link", ""))
            summary = html_fragment_to_text(getattr(entry, "summary", ""))
            published_at = normalize_space(
                getattr(entry, "published", "")
                or getattr(entry, "updated", "")
            )
            source_host = urlparse(link).netloc if link else "google-news"
            article_text = fetch_webpage_text(
                session=session,
                url=link,
                request_delay=request_delay,
            ) if link else ""
            text = article_text or summary or title
            if not text:
                continue

            records.append(
                Record(
                    id=stable_hash(f"google-news|{query}|{link}|{title}"),
                    source="news",
                    source_detail=source_host,
                    query=query,
                    url=link,
                    title=title,
                    text=text,
                    author="",
                    published_at=published_at,
                    collected_at=collected_at,
                )
            )

    return records


def collect_rss_feeds(
    session: requests.Session,
    feed_urls: Sequence[str],
    max_items_per_feed: int,
    request_delay: float,
) -> list[Record]:
    records: list[Record] = []
    collected_at = now_iso()

    for feed_url in feed_urls:
        parsed = feedparser.parse(feed_url)
        entries = parsed.entries[:max_items_per_feed]

        for entry in entries:
            title = normalize_space(getattr(entry, "title", ""))
            link = normalize_space(getattr(entry, "link", ""))
            summary = html_fragment_to_text(getattr(entry, "summary", ""))
            published_at = normalize_space(
                getattr(entry, "published", "")
                or getattr(entry, "updated", "")
            )
            source_host = urlparse(link).netloc if link else urlparse(feed_url).netloc
            article_text = fetch_webpage_text(
                session=session,
                url=link,
                request_delay=request_delay,
            ) if link else ""
            text = article_text or summary or title
            if not text:
                continue

            records.append(
                Record(
                    id=stable_hash(f"rss|{feed_url}|{link}|{title}"),
                    source="rss",
                    source_detail=source_host,
                    query="",
                    url=link,
                    title=title,
                    text=text,
                    author="",
                    published_at=published_at,
                    collected_at=collected_at,
                )
            )

    return records


def collect_direct_urls(
    session: requests.Session,
    urls: Sequence[str],
    request_delay: float,
) -> list[Record]:
    records: list[Record] = []
    collected_at = now_iso()
    for url in urls:
        text = fetch_webpage_text(session, url, request_delay=request_delay)
        if not text:
            continue
        host = urlparse(url).netloc
        records.append(
            Record(
                id=stable_hash(f"url|{url}"),
                source="web",
                source_detail=host,
                query="",
                url=url,
                title="",
                text=text,
                author="",
                published_at="",
                collected_at=collected_at,
            )
        )
    return records


def dedupe_records(records: Iterable[Record]) -> list[Record]:
    deduped: list[Record] = []
    seen: set[str] = set()
    for record in records:
        key = (record.url or "").strip().lower()
        if not key:
            key = stable_hash(f"{record.title}|{record.text[:160]}")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


class OllamaClient:
    def __init__(self, host: str, model: str, session: requests.Session) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.session = session

    def check_available(self) -> bool:
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models", [])
            names = [str(item.get("name", "")) for item in models]
            if self.model in names:
                return True
            base = self.model.split(":", 1)[0]
            return any(name == base or name.startswith(f"{base}:") for name in names)
        except (requests.RequestException, json.JSONDecodeError):
            return False

    def classify(self, title: str, text: str) -> dict[str, Any]:
        prompt = f"""
Classify the following text about vaccines and health behavior.
Return STRICT JSON only with keys:
stance, sentiment, confidence, reason

Allowed stance values:
- hesitant
- supportive
- neutral
- mixed
- unclear

Allowed sentiment values:
- negative
- neutral
- positive

Rules:
- confidence must be a number from 0 to 1.
- reason must be <= 25 words.
- Do not include extra keys.

Title: {title}
Text: {text}
"""
        payload = {
            "model": self.model,
            "prompt": prompt.strip(),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        try:
            response = self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            raw = response.json().get("response", "").strip()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            logging.warning("Ollama request failed: %s", exc)
            return self._fallback_label()

        parsed = self._parse_json(raw)
        if not parsed:
            return self._fallback_label()
        return parsed

    def _parse_json(self, raw: str) -> dict[str, Any]:
        obj: dict[str, Any] | None = None
        try:
            candidate = json.loads(raw)
            if isinstance(candidate, dict):
                obj = candidate
        except json.JSONDecodeError:
            pass

        if obj is None:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if match:
                try:
                    candidate = json.loads(match.group(0))
                    if isinstance(candidate, dict):
                        obj = candidate
                except json.JSONDecodeError:
                    obj = None

        if obj is None:
            return {}

        stance = str(obj.get("stance", "unclear")).strip().lower()
        if stance not in {"hesitant", "supportive", "neutral", "mixed", "unclear"}:
            stance = "unclear"

        sentiment = str(obj.get("sentiment", "neutral")).strip().lower()
        if sentiment not in {"negative", "neutral", "positive"}:
            sentiment = "neutral"

        try:
            confidence = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(max(confidence, 0.0), 1.0)

        reason = normalize_space(str(obj.get("reason", "")))
        if len(reason) > 200:
            reason = reason[:200]

        return {
            "stance": stance,
            "sentiment": sentiment,
            "confidence": confidence,
            "reason": reason,
        }

    @staticmethod
    def _fallback_label() -> dict[str, Any]:
        return {
            "stance": "unclear",
            "sentiment": "neutral",
            "confidence": 0.0,
            "reason": "",
        }


def apply_ollama_labels(
    records: list[Record],
    client: OllamaClient,
    max_chars: int,
) -> None:
    total = len(records)
    for idx, record in enumerate(records, start=1):
        text = normalize_space(record.text)
        if not text:
            continue
        snippet = text[:max_chars]
        label = client.classify(record.title, snippet)
        record.stance = str(label.get("stance", "unclear"))
        record.sentiment = str(label.get("sentiment", "neutral"))
        record.confidence = float(label.get("confidence", 0.0))
        record.reason = str(label.get("reason", ""))
        if idx % 10 == 0 or idx == total:
            logging.info("Labeled %s/%s records", idx, total)


def write_jsonl(path: Path, records: Sequence[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dataclasses.asdict(record), ensure_ascii=False))
            handle.write("\n")


def write_summary_csv(path: Path, records: Sequence[Record]) -> None:
    counts: dict[tuple[str, str, str, str], int] = {}
    for record in records:
        key = (
            record.source,
            record.source_detail,
            record.stance or "unlabeled",
            record.sentiment or "unlabeled",
        )
        counts[key] = counts.get(key, 0) + 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "source_detail", "stance", "sentiment", "count"])
        for key, count in sorted(counts.items()):
            writer.writerow([*key, count])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape Reddit + websites and classify vaccine/health hesitancy sentiment "
            "with local Ollama."
        )
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=DEFAULT_QUERIES,
        help="Search queries for Reddit and Google News RSS.",
    )
    parser.add_argument(
        "--subreddits",
        nargs="+",
        default=DEFAULT_SUBREDDITS,
        help="Reddit subreddits to search.",
    )
    parser.add_argument(
        "--max-reddit-per-query",
        type=int,
        default=40,
        help="Max Reddit posts per (subreddit, query).",
    )
    parser.add_argument(
        "--max-news-per-query",
        type=int,
        default=20,
        help="Max Google News RSS items per query.",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=30,
        help="How many recent days to search in Google News RSS queries.",
    )
    parser.add_argument(
        "--rss-feeds-file",
        type=Path,
        default=None,
        help="Optional file with one RSS feed URL per line.",
    )
    parser.add_argument(
        "--max-rss-items-per-feed",
        type=int,
        default=20,
        help="Max items scraped from each custom RSS feed.",
    )
    parser.add_argument(
        "--extra-urls-file",
        type=Path,
        default=None,
        help="Optional file with one webpage URL per line.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="Delay in seconds between web requests.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL.",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.1:8b",
        help="Local Ollama model name.",
    )
    parser.add_argument(
        "--ollama-max-chars",
        type=int,
        default=3500,
        help="Max text characters sent to Ollama per record.",
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Skip labeling and only export collected text.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    session = build_session()

    logging.info("Collecting Reddit posts...")
    records = collect_reddit(
        session=session,
        subreddits=args.subreddits,
        queries=args.queries,
        max_per_query=args.max_reddit_per_query,
        request_delay=args.request_delay,
    )

    logging.info("Collecting Google News RSS items...")
    records.extend(
        collect_google_news(
            session=session,
            queries=args.queries,
            days_back=args.days_back,
            max_per_query=args.max_news_per_query,
            request_delay=args.request_delay,
        )
    )

    if args.rss_feeds_file:
        logging.info("Collecting custom RSS feeds from %s...", args.rss_feeds_file)
        custom_feeds = load_lines(args.rss_feeds_file)
        records.extend(
            collect_rss_feeds(
                session=session,
                feed_urls=custom_feeds,
                max_items_per_feed=args.max_rss_items_per_feed,
                request_delay=args.request_delay,
            )
        )

    if args.extra_urls_file:
        logging.info("Collecting direct URLs from %s...", args.extra_urls_file)
        direct_urls = load_lines(args.extra_urls_file)
        records.extend(
            collect_direct_urls(
                session=session,
                urls=direct_urls,
                request_delay=args.request_delay,
            )
        )

    records = dedupe_records(records)
    logging.info("Collected %s unique records", len(records))

    raw_path = args.output_dir / "raw_records.jsonl"
    write_jsonl(raw_path, records)
    logging.info("Wrote raw records to %s", raw_path)

    if not args.skip_ollama and records:
        client = OllamaClient(
            host=args.ollama_url,
            model=args.ollama_model,
            session=session,
        )
        if client.check_available():
            logging.info("Labeling records with Ollama model '%s'...", args.ollama_model)
            apply_ollama_labels(records, client, max_chars=args.ollama_max_chars)
        else:
            logging.warning(
                "Ollama model '%s' not found at %s. Skipping labeling.",
                args.ollama_model,
                args.ollama_url,
            )

    labeled_path = args.output_dir / "labeled_records.jsonl"
    write_jsonl(labeled_path, records)
    logging.info("Wrote labeled records to %s", labeled_path)

    summary_path = args.output_dir / "summary.csv"
    write_summary_csv(summary_path, records)
    logging.info("Wrote summary to %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
