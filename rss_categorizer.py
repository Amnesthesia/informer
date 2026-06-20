#!/usr/bin/env python3
"""
RSS AI Categorizer

Reads an OPML file, fetches recent articles from every feed, sends them to an
AI provider (Claude or Gemini) for categorisation and summarisation, then
emits both a structured JSON result and a valid RSS 2.0 feed.

Usage:
    python rss_categorizer.py feeds.opml [options]

Environment variables (all overrideable by CLI flags):
    AI_PROVIDER   claude | gemini        (default: claude)
    FEED_MODE     daily  | weekly        (default: daily)
    ANTHROPIC_API_KEY                    required when provider=claude
    GOOGLE_API_KEY                       required when provider=gemini
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any
from xml.dom import minidom

import feedparser
from bs4 import BeautifulSoup

# ─── OPML parsing ─────────────────────────────────────────────────────────────

def parse_opml(path: str) -> list[dict]:
    """Return [{url, title}, ...] for every RSS/Atom outline in the OPML file."""
    tree = ET.parse(path)
    root = tree.getroot()
    feeds: list[dict] = []

    def _walk(el: ET.Element) -> None:
        for outline in el.findall("outline"):
            url = outline.get("xmlUrl") or outline.get("url")
            if url:
                feeds.append(
                    {
                        "url": url.strip(),
                        "title": (outline.get("title") or outline.get("text") or "").strip(),
                    }
                )
            _walk(outline)

    body = root.find("body")
    if body is not None:
        _walk(body)
    return feeds


# ─── Content helpers ──────────────────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")


def strip_html(raw: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    if not raw:
        return ""
    text = BeautifulSoup(raw, "html.parser").get_text(separator=" ", strip=True)
    return _WHITESPACE_RE.sub(" ", text).strip()


def best_text(entry: Any, max_chars: int = 2000) -> str:
    """Pick the richest text field from a feedparser entry."""
    # content[] has the full article body on many feeds
    if hasattr(entry, "content") and entry.content:
        raw = entry.content[0].get("value", "")
        if raw:
            return strip_html(raw)[:max_chars]
    summary = getattr(entry, "summary", "") or ""
    return strip_html(summary)[:max_chars]


def _struct_to_dt(t: Any) -> datetime | None:
    """Convert a feedparser time_struct to a tz-aware datetime."""
    if t is None:
        return None
    try:
        return datetime(*t[:6], tzinfo=timezone.utc)
    except Exception:
        return None


# ─── Article fetching ─────────────────────────────────────────────────────────

def fetch_articles(feeds: list[dict], cutoff: datetime, content_chars: int = 2000) -> list[dict]:
    """
    Fetch all feed entries published on or after *cutoff*.
    Returns a flat list of article dicts.
    """
    articles: list[dict] = []
    for feed_meta in feeds:
        url = feed_meta["url"]
        try:
            parsed = feedparser.parse(url)
        except Exception as exc:
            print(f"[warn] could not fetch {url}: {exc}", file=sys.stderr)
            continue

        feed_title = (
            getattr(parsed.feed, "title", None)
            or feed_meta["title"]
            or url
        )

        for entry in parsed.entries:
            pub = _struct_to_dt(
                getattr(entry, "published_parsed", None)
                or getattr(entry, "updated_parsed", None)
            )
            if pub is None or pub < cutoff:
                continue

            articles.append(
                {
                    "title": strip_html(getattr(entry, "title", "") or ""),
                    "url": getattr(entry, "link", "") or "",
                    "description": strip_html(getattr(entry, "summary", "") or ""),
                    "content": best_text(entry, content_chars),
                    "published": pub.isoformat(),
                    "source": feed_title,
                    "feed_url": url,
                }
            )

    return articles


# ─── AI provider abstraction ──────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a professional news analyst. "
    "Respond only with the JSON object requested — no prose, no markdown fences."
)

TASK_PROMPT = """\
Analyze the articles below and return a single JSON object with this exact schema:

{
  "categories": {
    "<CategoryName>": {
      "articles": [
        {
          "id": <integer, matches the id field in the input>,
          "summary": ["• <insight>", "• <insight>", ...],
          "relevance_tier": <1 | 2 | 3>,
          "tags": ["<topic>", ...]
        }
      ]
    }
  }
}

Rules:
• Derive categories organically from content — do not use a fixed list.
• Each article must appear in exactly one category.
• summary: 3-5 bullet points starting with "•", each ≤ 25 words.
• relevance_tier: 1 = must-read, 2 = worthwhile, 3 = low-priority.
• Within each category, order articles ascending by relevance_tier (most important first).
• tags: 3-6 concise topic keywords.

Articles:
"""


def _build_payload(articles: list[dict]) -> str:
    items = [
        {
            "id": i,
            "title": a["title"],
            "content": a["content"] or a["description"],
            "source": a["source"],
        }
        for i, a in enumerate(articles)
    ]
    return TASK_PROMPT + json.dumps(items, ensure_ascii=False, indent=2)


def _extract_json(text: str) -> dict:
    """Pull the first {...} JSON object out of a model response."""
    # Strip markdown fences if present
    clean = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean.strip())
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in model response")
    return json.loads(clean[start:end])


class AIProvider:
    """Base class — subclasses implement `_call(prompt) -> str`."""

    BATCH_SIZE = 30  # articles per API call

    def _call(self, prompt: str) -> str:
        raise NotImplementedError

    def analyze(self, articles: list[dict]) -> dict:
        """
        Send articles to the model in batches and merge the category results.
        Returns the canonical AI result dict: {"categories": {...}}.
        """
        merged: dict[str, list] = {}
        id_offset = 0

        for batch_start in range(0, len(articles), self.BATCH_SIZE):
            batch = articles[batch_start : batch_start + self.BATCH_SIZE]
            # Re-index within the batch so IDs start at 0 each time
            indexed_batch = [dict(a, _batch_idx=i) for i, a in enumerate(batch)]
            prompt = _build_payload(indexed_batch)

            try:
                raw = self._call(prompt)
                result = _extract_json(raw)
            except Exception as exc:
                print(f"[warn] AI batch {batch_start}–{batch_start+len(batch)} failed: {exc}", file=sys.stderr)
                result = {"categories": {}}

            for cat, cat_data in result.get("categories", {}).items():
                cat_articles = cat_data.get("articles", [])
                # Remap batch-local IDs back to global article indices
                for art in cat_articles:
                    art["id"] = art["id"] + id_offset
                merged.setdefault(cat, []).extend(cat_articles)

            id_offset += len(batch)

        return {"categories": {k: {"articles": v} for k, v in merged.items()}}


class ClaudeProvider(AIProvider):
    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        import anthropic  # lazy import so Gemini-only installs work

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("Error: ANTHROPIC_API_KEY is not set.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _call(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class GeminiProvider(AIProvider):
    def __init__(self, model: str = "gemini-2.0-flash") -> None:
        import google.generativeai as genai  # lazy import

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            sys.exit("Error: GOOGLE_API_KEY is not set.")
        genai.configure(api_key=api_key)
        self.model_obj = genai.GenerativeModel(
            model,
            system_instruction=SYSTEM_PROMPT,
        )

    def _call(self, prompt: str) -> str:
        response = self.model_obj.generate_content(prompt)
        return response.text


_PROVIDERS: dict[str, type[AIProvider]] = {
    "claude": ClaudeProvider,
    "gemini": GeminiProvider,
}


def make_provider(name: str) -> AIProvider:
    cls = _PROVIDERS.get(name.lower())
    if cls is None:
        sys.exit(f"Error: unknown provider '{name}'. Choose from: {', '.join(_PROVIDERS)}.")
    return cls()


# ─── Output assembly ──────────────────────────────────────────────────────────

def assemble_output(
    articles: list[dict],
    ai: dict,
    mode: str,
    start: datetime,
    end: datetime,
) -> dict:
    """Merge raw article metadata with AI annotations into the final JSON."""
    idx_map = {i: a for i, a in enumerate(articles)}
    categories: dict[str, list] = {}

    for cat_name, cat_data in ai.get("categories", {}).items():
        enriched = []
        for ai_art in sorted(
            cat_data.get("articles", []), key=lambda x: x.get("relevance_tier", 9)
        ):
            orig = idx_map.get(ai_art["id"])
            if orig is None:
                continue
            enriched.append(
                {
                    "title": orig["title"],
                    "summary": ai_art.get("summary", []),
                    "source": orig["source"],
                    "url": orig["url"],
                    "published": orig["published"],
                    "tags": {
                        "category": cat_name,
                        "relevance_tier": ai_art.get("relevance_tier", 3),
                        "source_feed": orig["source"],
                        "topics": ai_art.get("tags", []),
                    },
                }
            )
        if enriched:
            categories[cat_name] = enriched

    return {
        "mode": mode,
        "date_range": {
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
        },
        "generated_at": end.isoformat(),
        "article_count": len(articles),
        "categories": categories,
    }


# ─── RSS XML generation ────────────────────────────────────────────────────────

_TIER_STARS = {1: "★★★", 2: "★★", 3: "★"}
_TIER_LABEL = {1: "High", 2: "Medium", 3: "Low"}
_RFC2822 = "%a, %d %b %Y %H:%M:%S +0000"


def _rfc2822(iso: str, fallback: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime(_RFC2822)
    except Exception:
        return fallback


def _el(doc: minidom.Document, parent: minidom.Element, tag: str, text: str = "") -> minidom.Element:
    """Append a simple text element to *parent* and return it."""
    el = doc.createElement(tag)
    if text:
        el.appendChild(doc.createTextNode(text))
    parent.appendChild(el)
    return el


def _html_el(
    doc: minidom.Document, parent: minidom.Element, tag: str, html: str
) -> minidom.Element:
    """Append an element whose text content is wrapped in a CDATA section."""
    el = doc.createElement(tag)
    el.appendChild(doc.createCDATASection(html))
    parent.appendChild(el)
    return el


def build_rss(output: dict) -> str:
    """Convert the assembled JSON output into a pretty-printed RSS 2.0 string."""
    impl = minidom.getDOMImplementation()
    doc = impl.createDocument(None, "rss", None)
    rss = doc.documentElement
    rss.setAttribute("version", "2.0")
    rss.setAttribute("xmlns:dc", "http://purl.org/dc/elements/1.1/")
    rss.setAttribute("xmlns:atom", "http://www.w3.org/2005/Atom")

    channel = doc.createElement("channel")
    rss.appendChild(channel)

    mode = output["mode"]
    dr = output["date_range"]
    now_str = datetime.now(timezone.utc).strftime(_RFC2822)
    feed_title = f"AI-Categorised RSS Digest ({mode}: {dr['start']} – {dr['end']})"

    _el(doc, channel, "title", feed_title)
    _el(doc, channel, "description", f"{output['article_count']} articles processed across {len(output['categories'])} categories.")
    _el(doc, channel, "link", "https://github.com/amnesthesia/informer")
    _el(doc, channel, "lastBuildDate", now_str)
    _el(doc, channel, "generator", "rss_categorizer / AI-powered")

    # ── 1. Category digest entries (one per category) ────────────────────────
    for cat_name, articles in output["categories"].items():
        item = doc.createElement("item")
        channel.appendChild(item)

        _el(doc, item, "title", f"[{cat_name}] — {mode.capitalize()} digest {dr['start']}")
        _el(doc, item, "guid", f"digest:{cat_name}:{dr['start']}")
        _el(doc, item, "pubDate", now_str)
        _el(doc, item, "category", cat_name)
        _el(doc, item, "dc:creator", "AI Categorizer")

        lines = [f"<h2>{cat_name}</h2>", "<ol>"]
        for art in articles:
            tier = art["tags"]["relevance_tier"]
            stars = _TIER_STARS.get(tier, "")
            bullets = "".join(f"<li>{b}</li>" for b in art.get("summary", []))
            lines.append(
                f'<li>'
                f'{stars} <a href="{art["url"]}">{art["title"]}</a>'
                f' &mdash; <em>{art["source"]}</em>'
                f"<ul>{bullets}</ul>"
                f"</li>"
            )
        lines.append("</ol>")
        _html_el(doc, item, "description", "\n".join(lines))

    # ── 2. Individual article entries (sorted by relevance tier ascending) ───
    flat: list[tuple[int, str, dict]] = []
    for cat_name, articles in output["categories"].items():
        for art in articles:
            flat.append((art["tags"]["relevance_tier"], cat_name, art))
    flat.sort(key=lambda t: t[0])

    for tier, cat_name, art in flat:
        item = doc.createElement("item")
        channel.appendChild(item)

        _el(doc, item, "title", art["title"])
        _el(doc, item, "link", art["url"])
        _el(doc, item, "guid", art["url"])
        _el(doc, item, "pubDate", _rfc2822(art["published"], now_str))
        _el(doc, item, "source", art["source"])
        _el(doc, item, "dc:creator", art["source"])

        # Category tags: detected category, relevance tier, source feed
        _el(doc, item, "category", cat_name)
        _el(doc, item, "category", f"relevance:tier{tier}")
        _el(doc, item, "category", f"source:{art['source']}")
        for topic in art["tags"].get("topics", []):
            _el(doc, item, "category", topic)

        bullets = "".join(f"<li>{b}</li>" for b in art.get("summary", []))
        desc_html = (
            f"<p>"
            f"<strong>Category:</strong> {cat_name} &nbsp;|&nbsp; "
            f"<strong>Relevance:</strong> {_TIER_LABEL.get(tier, '')} ({_TIER_STARS.get(tier, '')}) &nbsp;|&nbsp; "
            f"<strong>Source:</strong> {art['source']}"
            f"</p>"
            f"<ul>{bullets}</ul>"
            f'<p><a href="{art["url"]}">Read full article →</a></p>'
        )
        _html_el(doc, item, "description", desc_html)

    return doc.toprettyxml(indent="  ", encoding=None)


# ─── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch, categorise, and summarise RSS feeds with AI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("opml", help="Path to the OPML feed list")
    p.add_argument(
        "--provider",
        default=os.environ.get("AI_PROVIDER", "claude"),
        choices=list(_PROVIDERS),
        help="AI provider  [env: AI_PROVIDER]",
    )
    p.add_argument(
        "--mode",
        default=os.environ.get("FEED_MODE", "daily"),
        choices=["daily", "weekly"],
        help="Article date window  [env: FEED_MODE]",
    )
    p.add_argument(
        "--content-chars",
        type=int,
        default=2000,
        metavar="N",
        help="Max characters of article body sent to the AI per article",
    )
    p.add_argument("--json-out", metavar="FILE", help="Write JSON result to FILE (default: stdout)")
    p.add_argument("--rss-out", metavar="FILE", help="Write RSS XML to FILE (default: not written)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Date window ──────────────────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    if args.mode == "weekly":
        cutoff = now - timedelta(days=7)
    else:
        # Start of today in UTC
        cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)

    print(
        f"[info] mode={args.mode}  window={cutoff.date()} → {now.date()}"
        f"  provider={args.provider}",
        file=sys.stderr,
    )

    # ── Fetch ─────────────────────────────────────────────────────────────────
    feeds = parse_opml(args.opml)
    print(f"[info] {len(feeds)} feed(s) found in OPML", file=sys.stderr)

    articles = fetch_articles(feeds, cutoff, args.content_chars)
    print(f"[info] {len(articles)} article(s) in the date window", file=sys.stderr)

    if not articles:
        print("[warn] No articles found — nothing to process.", file=sys.stderr)
        sys.exit(0)

    # ── AI analysis ───────────────────────────────────────────────────────────
    provider = make_provider(args.provider)
    print(f"[info] Sending {len(articles)} articles to {args.provider} …", file=sys.stderr)
    ai_result = provider.analyze(articles)

    # ── Assemble JSON output ──────────────────────────────────────────────────
    output = assemble_output(articles, ai_result, args.mode, cutoff, now)

    json_str = json.dumps(output, indent=2, ensure_ascii=False)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            fh.write(json_str)
        print(f"[info] JSON written to {args.json_out}", file=sys.stderr)
    else:
        print(json_str)

    # ── Build RSS ─────────────────────────────────────────────────────────────
    if args.rss_out:
        rss_str = build_rss(output)
        with open(args.rss_out, "w", encoding="utf-8") as fh:
            fh.write(rss_str)
        print(f"[info] RSS written to {args.rss_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
