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


def _hook_and_prose_from_html(html: str) -> tuple[str, str]:
    """Extract (one_line_hook, prose_summary) from an individual article description."""
    if not html:
        return "", ""
    soup = BeautifulSoup(html, "html.parser")
    hook = ""
    prose = ""
    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        if "Category:" in text or "Relevance:" in text:
            continue
        if "Read" in text and "article" in text:
            continue
        if not hook:
            em = p.find("em")
            if em:
                hook = em.get_text(strip=True)
                continue
        if text and not prose:
            prose = text
    return hook, prose


def _synthesis_from_digest_html(html: str) -> str:
    """Extract the category synthesis paragraph (the <p> before the <ul>)."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    p = soup.find("p")
    return p.get_text(" ", strip=True) if p else ""


def _articles_from_digest_html(html: str, cat_name: str) -> list[dict]:
    """
    Parse the HTML description of a category digest item into a list of article dicts
    matching the shape expected by render_markdown().

    The digest HTML structure (produced by build_rss) is:
        <h2>Category</h2>
        <p>Synthesis paragraph…</p>
        <ul>
          <li>
            <strong>★★ <a href="URL">Title</a></strong> — <em>Source</em><br/>
            <em>One-line hook…</em><br/>
            Prose summary…<br/>
            <a href="URL">Read more →</a>
          </li>
          ...
        </ul>
    """
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    ul = soup.find("ul")
    if not ul:
        return []

    articles = []
    for li in ul.find_all("li", recursive=False):
        a = li.find("a")
        if not a:
            continue
        title = a.get_text(strip=True)
        url = a.get("href", "")

        # First <em> is the source; second <em> is the one-line hook
        ems = li.find_all("em")
        source = ems[0].get_text(strip=True) if ems else ""
        hook = ems[1].get_text(strip=True) if len(ems) > 1 else ""

        # Stars in the text determine relevance tier
        full_text = li.get_text(" ")
        tier = 3
        if "★★★" in full_text:
            tier = 1
        elif "★★" in full_text:
            tier = 2

        # Prose summary: strip strong, a, em → remaining text
        for tag in li.find_all(["strong", "a", "em"]):
            tag.decompose()
        prose = li.get_text(" ", strip=True).strip("— ").strip()

        articles.append(
            {
                "title": title,
                "url": url,
                "source": source,
                "published": "",
                "category": cat_name,
                "tier": tier,
                "topics": [],
                "hook": hook,
                "prose": prose,
            }
        )
    return articles


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


def parse_feed(path: str) -> tuple[dict, list[dict], dict[str, str]]:
    """
    Parse an RSS file produced by rss_categorizer.py.

    Returns:
        meta     — channel-level metadata dict
        articles — list of article dicts
        synths   — map of category name → synthesis paragraph

    When the feed contains only category digest items (category-only mode, the
    default), falls back to extracting articles from the digest HTML so the
    markdown renderer has data to work with.
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
    digest_items: list[ET.Element] = []

    for item in channel.findall("item"):
        guid = item.findtext("guid", "")
        if guid.startswith("digest:"):
            digest_items.append(item)
            continue

        category, tier, source, topics = _parse_categories(item)
        desc_el = item.find("description")
        hook, prose = _hook_and_prose_from_html(desc_el.text if desc_el is not None else "")

        articles.append(
            {
                "title": item.findtext("title", "").strip(),
                "url": item.findtext("link", "").strip(),
                "source": source or item.findtext("source", "").strip(),
                "published": _rfc2822_to_date(item.findtext("pubDate", "")),
                "category": category or "Uncategorised",
                "tier": tier,
                "topics": topics,
                "hook": hook,
                "prose": prose,
            }
        )

    # Category-only feed — parse articles out of digest item descriptions
    synths: dict[str, str] = {}
    if not articles:
        for item in digest_items:
            cat_name = next(
                (el.text for el in item.findall("category") if el.text), ""
            )
            desc_el = item.find("description")
            html = desc_el.text or ""
            synths[cat_name] = _synthesis_from_digest_html(html)
            articles.extend(_articles_from_digest_html(html, cat_name))

    return meta, articles, synths


# ─── Markdown rendering ───────────────────────────────────────────────────────

def render_markdown(
    meta: dict, articles: list[dict], synths: dict[str, str] | None = None
) -> str:
    if synths is None:
        synths = {}
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

        if synths.get(cat):
            lines += [synths[cat], ""]

        for art in arts:
            badge = _TIER_BADGE.get(art["tier"], "")

            lines += [f"### [{art['title']}]({art['url']})", ""]
            lines += [f"**{art['source']}** · {art['published']} · `{badge}`", ""]

            hook = art.get("hook", "")
            if hook:
                lines += [f"*{hook}*", ""]

            prose = art.get("prose", "")
            if prose:
                lines += [prose, ""]

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

    meta, articles, synths = parse_feed(args.feed)
    md = render_markdown(meta, articles, synths)

    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(md)


if __name__ == "__main__":
    main()
