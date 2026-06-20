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
import concurrent.futures
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

_TASK_PROMPT = """\
Analyze the articles below and return a single JSON object with this exact schema:

{
  "categories": {
    "<CategoryName>": {
      "synthesis": "<2-3 sentence editorial overview of this category: what is happening, what is notable, where articles agree or contradict, and what is most worth reading>",
      "articles": [
        {
          "id": <integer, matches the id field in the input>,
          "one_line_hook": "<one sentence hook — tells the reader why this piece is worth their time>",
          "summary": "<prose paragraph of 3-4 sentences covering the key insight, broader context, and main takeaway>",
          "relevance_tier": <1 | 2>
        }
      ]
    }
  }
}

Rules:
• Create at most 8 broad thematic categories. Group related articles together — do NOT create a separate category per article or per source. Aim for 3–15 articles per category.
• Each article must appear in exactly one category.
• relevance_tier: 1 = must-read, 2 = worthwhile. Omit tier-3 (low-priority) articles entirely — do not include them in the articles array at all.
• Within each category, order articles by relevance_tier ascending (1 first).
• synthesis: write about the category as a whole, not any single article. Mention notable pieces by name or theme. Note where sources agree or contradict each other.
• one_line_hook: a hook, not a summary — one sentence that makes the reader want to click.
• summary: complete sentences only, no bullet points.

Articles:
"""


_BATCH_CONTENT_CHARS = 800  # per-article content limit inside a batch request


def _build_payload(articles: list[dict]) -> str:
    items = [
        {
            "id": i,
            "title": a["title"],
            "content": (a["content"] or a["description"])[:_BATCH_CONTENT_CHARS],
            "source": a["source"],
        }
        for i, a in enumerate(articles)
    ]
    return _TASK_PROMPT + json.dumps(items, ensure_ascii=False, indent=2)


def _repair_info(text: str, up_to: int) -> tuple[int, str]:
    """
    Single-pass, string-literal-aware scan of text[:up_to].

    Returns (last_comma_pos, closing_suffix) where closing_suffix is the
    exact sequence of `]` / `}` needed to close all containers open at the
    moment of that comma — in correct nesting order (innermost first).
    """
    in_string = False
    escaped = False
    stack: list[str] = []       # "{" or "[" for each open container
    last_comma = -1
    comma_stack: list[str] = []

    for i, ch in enumerate(text[:up_to]):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == ",":
            last_comma = i
            comma_stack = stack.copy()
        elif ch == "{":
            stack.append("{")
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "[":
            stack.append("[")
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    closing = "".join("}" if c == "{" else "]" for c in reversed(comma_stack))
    return last_comma, closing


def _extract_json(text: str) -> dict:
    """
    Parse a JSON object from model output.

    Handles two common failure modes:
    - Markdown fences wrapping the JSON
    - Response truncated mid-object (output token limit hit): cuts at the last
      comma outside any string literal, then closes open brackets/braces using
      the depth recorded at that exact comma position.
    """
    clean = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean.strip())

    start = clean.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response")
    fragment = clean[start:]

    try:
        return json.loads(fragment)
    except json.JSONDecodeError as exc:
        cut, closing = _repair_info(fragment, exc.pos)
        if cut == -1:
            raise ValueError(f"Cannot repair JSON (no comma before error at pos {exc.pos})") from exc
        truncated = fragment[:cut]
        try:
            result = json.loads(truncated + closing)
            print(
                f"[warn] Repaired truncated JSON (trimmed at char {cut}, added {closing!r})",
                file=sys.stderr,
            )
            return result
        except json.JSONDecodeError as exc2:
            raise ValueError(f"Could not repair truncated JSON: {exc2}") from exc2


class AIProvider:
    """Base class — subclasses implement `_call(prompt) -> str`."""

    BATCH_SIZE = 15  # articles per API call — keeps response within 8 k output tokens

    def _call(self, prompt: str) -> str:
        raise NotImplementedError

    def analyze(self, articles: list[dict]) -> dict:
        """
        Send articles to the model in batches concurrently and merge the results.
        Returns the canonical AI result dict: {"categories": {...}}.
        """
        batches = [
            articles[i : i + self.BATCH_SIZE]
            for i in range(0, len(articles), self.BATCH_SIZE)
        ]
        jobs = [
            (i * self.BATCH_SIZE, batch, _build_payload(batch))
            for i, batch in enumerate(batches)
        ]

        def _run(id_offset: int, batch: list[dict], prompt: str) -> dict:
            try:
                raw = self._call(prompt)
                result = _extract_json(raw)
            except Exception as exc:
                print(
                    f"[warn] AI batch starting at article {id_offset} failed: {exc}",
                    file=sys.stderr,
                )
                result = {"categories": {}}
            for cat_data in result.get("categories", {}).values():
                for art in cat_data.get("articles", []):
                    art["id"] += id_offset
            return result

        merged: dict[str, dict] = {}
        with concurrent.futures.ThreadPoolExecutor() as pool:
            futures = [pool.submit(_run, off, b, p) for off, b, p in jobs]
            for fut in concurrent.futures.as_completed(futures):
                for cat, cat_data in fut.result().get("categories", {}).items():
                    if cat not in merged:
                        merged[cat] = {"synthesis": "", "articles": []}
                    merged[cat]["articles"].extend(cat_data.get("articles", []))
                    if not merged[cat]["synthesis"]:
                        merged[cat]["synthesis"] = cat_data.get("synthesis", "")

        return {"categories": merged}


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
    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        from google import genai  # lazy import — requires google-genai package

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            sys.exit("Error: GOOGLE_API_KEY is not set.")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _call(self, prompt: str) -> str:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=8192,
            ),
        )
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
    categories: dict[str, dict] = {}

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
                    "one_line_hook": ai_art.get("one_line_hook", ""),
                    "summary": ai_art.get("summary", ""),
                    "source": orig["source"],
                    "url": orig["url"],
                    "published": orig["published"],
                    "tags": {
                        "category": cat_name,
                        "relevance_tier": ai_art.get("relevance_tier", 2),
                        "source_feed": orig["source"],
                    },
                }
            )
        if enriched:
            categories[cat_name] = {
                "synthesis": cat_data.get("synthesis", ""),
                "articles": enriched,
            }

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


def build_rss(output: dict, include_articles: bool = False) -> str:
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
    date_label = dr["start"] if mode == "daily" else f"{dr['start']} to {dr['end']}"

    for cat_name, cat_data in output["categories"].items():
        articles = cat_data["articles"]
        synthesis = cat_data.get("synthesis", "")

        item = doc.createElement("item")
        channel.appendChild(item)

        _el(doc, item, "title", f"[{cat_name}] — {mode.capitalize()} digest {date_label}")
        _el(doc, item, "guid", f"digest:{cat_name}:{dr['start']}:{dr['end']}")
        _el(doc, item, "pubDate", now_str)
        _el(doc, item, "category", cat_name)
        _el(doc, item, "dc:creator", "AI Categorizer")

        lines = [f"<h2>{cat_name}</h2>"]
        if synthesis:
            lines.append(f"<p>{synthesis}</p>")
        lines.append("<ul>")
        for art in articles:
            tier = art["tags"]["relevance_tier"]
            stars = _TIER_STARS.get(tier, "")
            hook = art.get("one_line_hook", "")
            summary = art.get("summary", "")
            lines.append(
                f"<li>"
                f"<strong>{stars} <a href=\"{art['url']}\">{art['title']}</a></strong>"
                f" &mdash; <em>{art['source']}</em><br/>"
                f"<em>{hook}</em><br/>"
                f"{summary}<br/>"
                f"<a href=\"{art['url']}\">Read more →</a>"
                f"</li>"
            )
        lines.append("</ul>")
        _html_el(doc, item, "description", "\n".join(lines))

    # ── 2. Individual article entries (only when --include-articles is set) ──
    if include_articles:
        flat: list[tuple[int, str, dict]] = []
        for cat_name, cat_data in output["categories"].items():
            for art in cat_data["articles"]:
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

            _el(doc, item, "category", cat_name)
            _el(doc, item, "category", f"relevance:tier{tier}")
            _el(doc, item, "category", f"source:{art['source']}")

            hook = art.get("one_line_hook", "")
            summary = art.get("summary", "")
            desc_html = (
                f"<p>"
                f"<strong>Category:</strong> {cat_name} &nbsp;|&nbsp; "
                f"<strong>Relevance:</strong> {_TIER_LABEL.get(tier, '')} ({_TIER_STARS.get(tier, '')}) &nbsp;|&nbsp; "
                f"<strong>Source:</strong> {art['source']}"
                f"</p>"
                f"<p><em>{hook}</em></p>"
                f"<p>{summary}</p>"
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
    include_default = os.environ.get("INCLUDE_ARTICLES", "").lower() in ("1", "true", "yes")
    p.add_argument(
        "--include-articles",
        action="store_true",
        default=include_default,
        help="Append individual article entries below category digests  [env: INCLUDE_ARTICLES]",
    )
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
        rss_str = build_rss(output, include_articles=args.include_articles)
        with open(args.rss_out, "w", encoding="utf-8") as fh:
            fh.write(rss_str)
        print(f"[info] RSS written to {args.rss_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
