#!/usr/bin/env python3
"""
rss_to_markdown.py — converts rss_categorizer.py feed output to a readable Markdown digest.

Usage:
    python rss_to_markdown.py feed.rss
    python rss_to_markdown.py feed.rss --output digest.md
"""

from __future__ import annotations

import argparse
import re
import sys
from email.utils import parsedate_to_datetime
from pathlib import Path
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

# ─── Helpers ──────────────────────────────────────────────────────────────────

_TIER_BADGE = {1: "★★★ High", 2: "★★ Medium", 3: "★ Low"}


def _rfc2822_to_date(s: str) -> str:
    try:
        return parsedate_to_datetime(s).strftime("%Y-%m-%d")
    except Exception:
        return s or "unknown"


def _slug(text: str) -> str:
    """GitHub-flavoured markdown heading anchor."""
    return re.sub(r"[^\w\s-]", "", text.lower()).strip().replace(" ", "-")


def _bullets_from_html(html: str) -> list[str]:
    """Extract <li> text from an HTML fragment, stripping leading bullet chars."""
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for li in soup.find_all("li"):
        # Only grab direct-child lis (not nested ones inside category digests)
        if li.find_parent("li"):
            continue
        text = li.get_text(" ", strip=True).lstrip("•").strip()
        if text:
            results.append(text)
    return results


# ─── RSS parsing ──────────────────────────────────────────────────────────────

def _parse_categories(item: ET.Element) -> tuple[str, int, str, list[str]]:
    """
    Return (category, relevance_tier, source_feed, topic_tags) from <category> elements.

    Convention written by rss_categorizer.py:
        category[0]   = detected category name
        category[1]   = "relevance:tier<N>"
        category[2]   = "source:<Feed Name>"
        category[3+]  = topic keywords
    """
    cats = [el.text or "" for el in item.findall("category")]
    detected = ""
    tier = 3
    source = ""
    topics: list[str] = []

    for c in cats:
        if c.startswith("relevance:tier"):
            try:
                tier = int(c[-1])
            except ValueError:
                pass
        elif c.startswith("source:"):
            source = c[7:]
        elif c and not detected:
            detected = c
        elif c:
            topics.append(c)

    return detected, tier, source, topics


def parse_feed(path: str) -> tuple[dict, list[dict]]:
    """
    Parse an RSS file produced by rss_categorizer.py.

    Returns:
        meta     — channel-level metadata dict
        articles — list of article dicts (category digest items are skipped)
    """
    tree = ET.parse(path)
    channel = tree.getroot().find("channel")
    if channel is None:
        sys.exit(f"Error: no <channel> element found in {path}")

    meta = {
        "title": channel.findtext("title", ""),
        "description": channel.findtext("description", ""),
        "last_build": _rfc2822_to_date(channel.findtext("lastBuildDate", "")),
    }

    articles: list[dict] = []
    for item in channel.findall("item"):
        guid = item.findtext("guid", "")
        # Category digest summary entries are identified by their guid prefix
        if guid.startswith("digest:"):
            continue

        category, tier, source, topics = _parse_categories(item)

        desc_el = item.find("description")
        bullets = _bullets_from_html(desc_el.text if desc_el is not None else "")

        articles.append(
            {
                "title": item.findtext("title", "").strip(),
                "url": item.findtext("link", "").strip(),
                "source": source or item.findtext("source", "").strip(),
                "published": _rfc2822_to_date(item.findtext("pubDate", "")),
                "category": category or "Uncategorised",
                "tier": tier,
                "topics": topics,
                "bullets": bullets,
            }
        )

    return meta, articles


# ─── Markdown rendering ───────────────────────────────────────────────────────

def render_markdown(meta: dict, articles: list[dict]) -> str:
    if not articles:
        return f"# {meta['title']}\n\n*No articles found.*\n"

    # Group by category, sort each group by tier ascending (1 = most important first)
    groups: dict[str, list[dict]] = {}
    for art in articles:
        groups.setdefault(art["category"], []).append(art)
    for arts in groups.values():
        arts.sort(key=lambda a: a["tier"])

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        f"# {meta['title']}",
        "",
        f"*{meta['description']}*  ",
        f"*Generated: {meta['last_build']} · {len(articles)} article{'s' if len(articles) != 1 else ''}"
        f" across {len(groups)} categor{'ies' if len(groups) != 1 else 'y'}*",
        "",
        "---",
        "",
    ]

    # ── Table of contents ─────────────────────────────────────────────────────
    lines += ["## Contents", ""]
    for cat in sorted(groups):
        count = len(groups[cat])
        lines.append(f"- [{cat}](#{_slug(cat)}) — {count} article{'s' if count != 1 else ''}")
    lines += ["", "---", ""]

    # ── Category sections ─────────────────────────────────────────────────────
    for cat in sorted(groups):
        arts = groups[cat]
        lines += [f"## {cat}", ""]

        for art in arts:
            badge = _TIER_BADGE.get(art["tier"], "")

            # Article heading as a link
            lines += [f"### [{art['title']}]({art['url']})", ""]

            # Meta line
            lines += [
                f"**{art['source']}** · {art['published']} · `{badge}`",
                "",
            ]

            # Bullet summary
            if art["bullets"]:
                for b in art["bullets"]:
                    lines.append(f"- {b}")
                lines.append("")

            # Topic tags
            if art["topics"]:
                lines += [" ".join(f"`{t}`" for t in art["topics"]), ""]

            lines += ["---", ""]

    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert an rss_categorizer.py feed.rss into readable Markdown.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("feed", help="Path to the RSS file (feed.rss)")
    p.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Write Markdown to FILE instead of stdout",
    )
    args = p.parse_args()

    if not Path(args.feed).exists():
        sys.exit(f"Error: {args.feed} not found.")

    meta, articles = parse_feed(args.feed)
    md = render_markdown(meta, articles)

    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(md)


if __name__ == "__main__":
    main()
