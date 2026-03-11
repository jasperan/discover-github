"""
Multi-platform content formatter for discover-github.

Generates platform-native content from a single RepoAnalysis:
- Twitter/X: Thread with hook + detail tweets + CTA
- LinkedIn: Professional tone, hashtags, question hook
- Newsletter: Curated digest with categories and deep-dives
- Blog: Long-form with code context and comparisons
- Discord/Slack: Embed-style compact summary
"""

import textwrap
from datetime import datetime, timezone

from analyzer import RepoAnalysis
from github_client import RepoMetadata
from scoring import ScoreBreakdown


def format_twitter_thread(
    meta: RepoMetadata,
    analysis: RepoAnalysis,
    score: ScoreBreakdown,
) -> list[str]:
    """
    Generate a Twitter/X thread (list of tweets, each <=280 chars).
    Structure: hook → what it does → why it matters → link.
    """
    tweets = []

    # Tweet 1: Hook
    hook = analysis.social_hooks[0] if analysis.social_hooks else analysis.one_liner
    hook_tweet = _truncate(hook, 250)
    if meta.language:
        hook_tweet += f"\n\n#{meta.language}"
    tweets.append(_truncate(hook_tweet, 280))

    # Tweet 2: What it does
    what = analysis.problem_solved or analysis.one_liner
    tweets.append(_truncate(f"What it does:\n\n{what}", 280))

    # Tweet 3: Why it's different
    if analysis.whats_novel:
        novel_tweet = f"What's novel:\n\n{analysis.whats_novel}"
        tweets.append(_truncate(novel_tweet, 280))

    # Tweet 4: Key features as bullet points
    if analysis.key_features:
        features = "\n".join(f"• {f}" for f in analysis.key_features[:4])
        tweets.append(_truncate(f"Key features:\n\n{features}", 280))

    # Tweet 5: Stats + link
    stats = f"⭐ {_format_number(meta.stars)} stars"
    if meta.forks > 0:
        stats += f" | 🍴 {_format_number(meta.forks)} forks"
    if meta.contributor_count > 1:
        stats += f" | 👥 {meta.contributor_count} contributors"

    link_tweet = f"{stats}\n\nBreakout Score: {score.total:.0f}/100 (Tier {score.tier})\n\n{meta.url}"
    tweets.append(_truncate(link_tweet, 280))

    return tweets


def format_linkedin_post(
    meta: RepoMetadata,
    analysis: RepoAnalysis,
    score: ScoreBreakdown,
) -> str:
    """
    Generate a LinkedIn post (~1300 chars, professional tone).
    Structure: question hook → insight → details → CTA → hashtags.
    """
    # Question hook
    if analysis.social_hooks:
        hook = analysis.social_hooks[0]
        if not hook.endswith("?"):
            hook = f"Have you seen this? {hook}"
    else:
        hook = f"Discovered a project worth watching: {meta.full_name}"

    # Body
    body_parts = []

    if analysis.problem_solved:
        body_parts.append(analysis.problem_solved)

    if analysis.whats_novel:
        body_parts.append(f"What makes it stand out: {analysis.whats_novel}")

    if analysis.key_features:
        features = "\n".join(f"→ {f}" for f in analysis.key_features[:3])
        body_parts.append(f"Key capabilities:\n{features}")

    if analysis.potential_impact:
        body_parts.append(analysis.potential_impact)

    body = "\n\n".join(body_parts)

    # Stats line
    stats = f"📊 {_format_number(meta.stars)} stars"
    if score.tier in ("S", "A"):
        stats += f" | Breakout Tier {score.tier}"

    # Hashtags from categories + language
    hashtags = []
    if meta.language:
        hashtags.append(f"#{meta.language}")
    for cat in analysis.categories[:2]:
        tag = cat.replace("/", "").replace("-", "").replace(" ", "")
        hashtags.append(f"#{tag}")
    hashtags.extend(["#OpenSource", "#GitHub"])
    hashtag_str = " ".join(hashtags[:5])

    post = f"""{hook}

{body}

{stats}
🔗 {meta.url}

{hashtag_str}"""

    return _truncate(post, 3000)


def format_newsletter_entry(
    meta: RepoMetadata,
    analysis: RepoAnalysis,
    score: ScoreBreakdown,
) -> str:
    """
    Generate a newsletter digest entry for one repo.
    Designed to be concatenated with other entries for a full newsletter.
    """
    tier_emoji = {"S": "🔥", "A": "⭐", "B": "✅", "C": "📌", "D": "📎"}.get(score.tier, "📎")

    features = ""
    if analysis.key_features:
        features = "\n".join(f"  - {f}" for f in analysis.key_features[:3])
        features = f"\n{features}"

    red_flags_text = ""
    if analysis.red_flags:
        red_flags_text = f"\n  ⚠️ {'; '.join(analysis.red_flags[:2])}"

    entry = f"""### {tier_emoji} [{meta.full_name}]({meta.url}) — {analysis.one_liner}

**{meta.language or 'Multi-language'}** | ⭐ {_format_number(meta.stars)} | Score: {score.total:.0f}/100

{analysis.problem_solved}

**Why it's different:** {analysis.whats_novel}
{features}{red_flags_text}

---"""
    return entry


def format_discord_embed(
    meta: RepoMetadata,
    analysis: RepoAnalysis,
    score: ScoreBreakdown,
) -> dict:
    """
    Generate a Discord/Slack embed-style summary (JSON-compatible dict).
    Can be sent via webhook or bot API.
    """
    tier_colors = {
        "S": 0xFF4500,  # Red-orange
        "A": 0xFFA500,  # Orange
        "B": 0x32CD32,  # Green
        "C": 0x4169E1,  # Blue
        "D": 0x808080,  # Gray
    }

    fields = [
        {"name": "Language", "value": meta.language or "N/A", "inline": True},
        {"name": "Stars", "value": _format_number(meta.stars), "inline": True},
        {"name": "Score", "value": f"{score.total:.0f}/100 ({score.tier})", "inline": True},
    ]

    if analysis.whats_novel:
        fields.append({
            "name": "What's Novel",
            "value": _truncate(analysis.whats_novel, 200),
            "inline": False,
        })

    if analysis.who_should_care:
        fields.append({
            "name": "Who Should Care",
            "value": _truncate(analysis.who_should_care, 150),
            "inline": False,
        })

    return {
        "title": meta.full_name,
        "description": analysis.one_liner,
        "url": meta.url,
        "color": tier_colors.get(score.tier, 0x808080),
        "fields": fields,
        "footer": {
            "text": f"Discovered by discover-github | {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        },
    }


def format_blog_intro(
    meta: RepoMetadata,
    analysis: RepoAnalysis,
    score: ScoreBreakdown,
    similar_repos: list[dict] | None = None,
) -> str:
    """
    Generate an 800-word blog post introduction with technical depth.
    """
    # Title
    title = f"# {meta.full_name}: {analysis.one_liner}\n"

    # Meta
    meta_line = (
        f"*{meta.language or 'Multi-language'} · "
        f"⭐ {_format_number(meta.stars)} stars · "
        f"Breakout Score: {score.total:.0f}/100 (Tier {score.tier}) · "
        f"[GitHub]({meta.url})*\n"
    )

    # Opening hook
    hook = analysis.social_hooks[0] if analysis.social_hooks else analysis.one_liner
    opening = f"\n{hook}\n"

    # Problem section
    problem = f"\n## The Problem\n\n{analysis.problem_solved}\n"

    # What's novel
    novel = f"\n## What Makes It Different\n\n{analysis.whats_novel}\n"

    # Features
    features_section = ""
    if analysis.key_features:
        features_list = "\n".join(f"- **{f}**" for f in analysis.key_features)
        features_section = f"\n## Key Features\n\n{features_list}\n"

    # Competitive landscape
    competitive = ""
    if analysis.competitive_landscape:
        competitive = f"\n## How It Compares\n\n{analysis.competitive_landscape}\n"

    if similar_repos:
        competitive += "\n**Related projects in our database:**\n"
        for s in similar_repos[:3]:
            competitive += f"- [{s['full_name']}](https://github.com/{s['full_name']}) — {s['description']} ({s['similarity']:.0%} similar)\n"

    # Impact
    impact = ""
    if analysis.potential_impact:
        impact = f"\n## Why This Matters\n\n{analysis.potential_impact}\n"

    # Verdict
    flags_text = ""
    if analysis.red_flags:
        flags_text = "\n**Watch out for:** " + ", ".join(analysis.red_flags)

    audience = ""
    if analysis.who_should_care:
        audience = f"\n**Best for:** {analysis.who_should_care}"

    verdict = f"\n## Verdict\n\nBreakout Score: **{score.total:.0f}/100** (Tier {score.tier}){audience}{flags_text}\n"

    return title + meta_line + opening + problem + novel + features_section + competitive + impact + verdict


def format_trend_synthesis_newsletter(synthesis: dict, date: str | None = None) -> str:
    """
    Format cross-repo trend synthesis into a newsletter header.
    """
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    header = f"# GitHub Trending Intelligence — {date}\n\n"

    if synthesis.get("overall_narrative"):
        header += f"**{synthesis['overall_narrative']}**\n\n"

    if synthesis.get("surprise_factor"):
        header += f"💡 *Surprise finding: {synthesis['surprise_factor']}*\n\n"

    header += "---\n\n## Emerging Trends\n\n"

    for trend in synthesis.get("meta_trends", []):
        repos = ", ".join(trend.get("repos", []))
        header += f"### {trend['trend']}\n\n"
        header += f"{trend['description']}\n\n"
        header += f"**Repos:** {repos}\n"
        header += f"**Significance:** {trend.get('significance', '')}\n\n"

    return header


# --- Helpers ---

def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len, adding ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _format_number(n: int) -> str:
    """Format large numbers: 1500 → 1.5k, 1500000 → 1.5M."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)
