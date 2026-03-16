#!/usr/bin/env python3
"""
discover-github: Intelligent GitHub trending repository discovery engine.

Pipeline:
  1. Discover trending repos via GitHub API (multi-strategy)
  2. Enrich with metadata (contributors, commits, CI, README)
  3. Generate embeddings for semantic dedup + vector search
  4. Score each repo with the breakout scoring engine
  5. Deduplicate against previously seen repos (Oracle AI Vector Search)
  6. Analyze with structured LLM prompts ("Why This Matters")
  7. Generate multi-platform content (Twitter, LinkedIn, Newsletter, Blog)
  8. Persist everything to Oracle DB for trend tracking

Usage:
    python discover.py                          # Full pipeline, default settings
    python discover.py --language python        # Filter by language
    python discover.py --strategy hot           # Use 'hot recent' strategy
    python discover.py --limit 10               # Discover top 10 only
    python discover.py --skip-analysis          # Skip LLM analysis (fast mode)
    python discover.py --output-dir ./outputs   # Custom output directory
    python discover.py --format linkedin        # Generate only LinkedIn content
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from github_client import GitHubClient, RepoMetadata
from db import DiscoverDB
from scoring import compute_breakout_score, rank_repos, ScoreBreakdown
from embeddings import generate_embedding
from analyzer import RepoAnalyzer, RepoAnalysis
from content_formatter import (
    format_twitter_thread,
    format_linkedin_post,
    format_newsletter_entry,
    format_discord_embed,
    format_blog_intro,
    format_trend_synthesis_newsletter,
)

logger = logging.getLogger("discover-github")


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        logger.error(f"Config file not found: {path}")
        logger.error("Copy config.example.yaml to config.yaml and fill in your settings.")
        sys.exit(1)

    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_pipeline(args):
    """Execute the full discovery pipeline."""
    config = load_config(args.config)
    setup_logging(args.verbose)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Initialize components ---
    logger.info("Initializing components...")

    gh = GitHubClient(token=config.get("github_token"))

    db = DiscoverDB(
        user=config.get("db_user", "discover_github"),
        password=config.get("db_password", "discover_github"),
        dsn=config.get("db_dsn", "localhost:1525/FREEPDB1"),
    )
    db.initialize_schema()

    logger.info(f"Database has {db.get_repo_count()} repos stored")

    # --- Step 2: Discover repos ---
    strategy = args.strategy
    language = args.language
    limit = args.limit

    logger.info(f"Strategy: {strategy} | Language: {language or 'all'} | Limit: {limit}")

    if strategy == "trending":
        repos = gh.discover_trending(
            language=language, since_days=1, limit=limit, min_stars=10,
        )
    elif strategy == "hot":
        repos = gh.discover_hot_recent(
            days_back=7, limit=limit, min_stars=50, language=language,
        )
    elif strategy == "surging":
        repos = gh.discover_surging(
            days_back=30, limit=limit, min_stars=100, language=language,
        )
    elif strategy == "all":
        # Combine all strategies, deduplicate by full_name
        seen = set()
        repos = []
        per_strategy = max(limit // 3, 10)

        for strat_repos in [
            gh.discover_trending(language=language, limit=per_strategy),
            gh.discover_hot_recent(language=language, limit=per_strategy),
            gh.discover_surging(language=language, limit=per_strategy),
        ]:
            for r in strat_repos:
                if r.full_name not in seen:
                    seen.add(r.full_name)
                    repos.append(r)
        repos = repos[:limit]
    else:
        logger.error(f"Unknown strategy: {strategy}")
        sys.exit(1)

    if not repos:
        logger.warning("No repositories discovered. Check your GitHub token and filters.")
        return

    logger.info(f"Discovered {len(repos)} repositories")

    # --- Step 3: Enrich with metadata ---
    logger.info("Enriching repositories with metadata...")
    for i, meta in enumerate(repos):
        logger.info(f"  [{i+1}/{len(repos)}] Enriching {meta.full_name}...")
        gh.enrich_repo(meta)
        if i < len(repos) - 1:
            time.sleep(1)  # Rate limit: avoid hitting GitHub API limits

    # --- Step 4: Generate embeddings + deduplicate ---
    logger.info("Generating embeddings and checking for duplicates...")
    novel_repos = []
    skipped_dupes = 0

    for meta in repos:
        embed_text = f"{meta.description or ''} {meta.readme_content or ''}"
        embedding = generate_embedding(embed_text)

        # Check for semantic duplicates in DB
        dupe = db.is_duplicate(embedding, threshold=0.92, exclude_full_name=meta.full_name)
        if dupe:
            logger.info(f"  Skipping {meta.full_name} (too similar to {dupe})")
            skipped_dupes += 1
            # Still upsert to update metrics, but don't analyze
            db.upsert_repo(meta, embedding)
            continue

        # Persist to DB
        repo_id = db.upsert_repo(meta, embedding)
        meta._db_id = repo_id  # Attach for later use
        meta._embedding = embedding
        novel_repos.append(meta)

    logger.info(
        f"Novel repos: {len(novel_repos)} | "
        f"Duplicates skipped: {skipped_dupes}"
    )

    if not novel_repos:
        logger.info("All discovered repos are already in the database. Nothing new to analyze.")
        return

    # --- Step 5: Score repos ---
    logger.info("Computing breakout scores...")
    star_velocities = {}
    for meta in novel_repos:
        vel = db.get_star_velocity(meta.full_name)
        if vel is not None:
            star_velocities[meta.full_name] = vel

    ranked = rank_repos(novel_repos, star_velocities)

    # Record snapshots
    for meta, score in ranked:
        db.record_snapshot(meta._db_id, score.total)

    # Log rankings
    logger.info("=" * 70)
    logger.info("BREAKOUT RANKINGS")
    logger.info("=" * 70)
    for i, (meta, score) in enumerate(ranked):
        logger.info(
            f"  #{i+1} [{score.tier}] {score.total:5.1f}  "
            f"⭐{meta.stars:>7,}  {meta.full_name}"
        )
        if score.flags:
            logger.info(f"         Flags: {', '.join(score.flags)}")
    logger.info("=" * 70)

    # --- Step 6: LLM Analysis ---
    analyses = []
    if not args.skip_analysis:
        logger.info("Running 'Why This Matters' LLM analysis...")
        try:
            llm = RepoAnalyzer(config_path=args.config)
        except Exception as e:
            logger.error(f"Failed to initialize LLM analyzer: {e}")
            logger.info("Skipping analysis. Check your OCI config in config.yaml.")
            args.skip_analysis = True

        if not args.skip_analysis:
            for i, (meta, score) in enumerate(ranked):
                logger.info(f"  [{i+1}/{len(ranked)}] Analyzing {meta.full_name}...")

                # Find similar repos for competitive context
                similar = db.find_similar(
                    meta._embedding, limit=3, min_similarity=0.5,
                    exclude_full_name=meta.full_name,
                )

                try:
                    analysis = llm.analyze_repo(meta, score, similar)
                    analyses.append(analysis)

                    # Save categories
                    if analysis.categories:
                        cats = [{"category": c, "confidence": 1.0} for c in analysis.categories]
                        db.save_categories(meta._db_id, cats)

                    logger.info(f"    → {analysis.one_liner[:80]}")
                except Exception as e:
                    logger.warning(f"    Analysis failed: {e}")
                    analyses.append(RepoAnalysis(
                        repo_full_name=meta.full_name,
                        one_liner=meta.description or meta.full_name,
                    ))

            # Trend synthesis
            if len(analyses) >= 3:
                logger.info("Generating cross-repo trend synthesis...")
                try:
                    scores_list = [s for _, s in ranked]
                    synthesis = llm.synthesize_trends(analyses, scores_list)
                except Exception as e:
                    logger.warning(f"Trend synthesis failed: {e}")
                    synthesis = {}
            else:
                synthesis = {}
    else:
        # Create minimal analyses from metadata
        for meta, score in ranked:
            analyses.append(RepoAnalysis(
                repo_full_name=meta.full_name,
                one_liner=meta.description or meta.full_name,
            ))
        synthesis = {}

    # --- Step 7: Generate content ---
    logger.info("Generating multi-platform content...")
    formats = args.format.split(",") if args.format != "all" else [
        "twitter", "linkedin", "newsletter", "discord", "blog",
    ]

    # Individual repo content
    for (meta, score), analysis in zip(ranked, analyses):
        repo_dir = output_dir / meta.full_name.replace("/", "__")
        repo_dir.mkdir(parents=True, exist_ok=True)

        if "twitter" in formats:
            thread = format_twitter_thread(meta, analysis, score)
            content = "\n\n---\n\n".join(thread)
            (repo_dir / "twitter_thread.txt").write_text(content)
            db.save_content(meta._db_id, "twitter_thread", content, "twitter")

        if "linkedin" in formats:
            post = format_linkedin_post(meta, analysis, score)
            (repo_dir / "linkedin_post.txt").write_text(post)
            db.save_content(meta._db_id, "linkedin_post", post, "linkedin")

        if "newsletter" in formats:
            entry = format_newsletter_entry(meta, analysis, score)
            (repo_dir / "newsletter_entry.md").write_text(entry)
            db.save_content(meta._db_id, "newsletter_entry", entry, "newsletter")

        if "discord" in formats:
            embed = format_discord_embed(meta, analysis, score)
            (repo_dir / "discord_embed.json").write_text(json.dumps(embed, indent=2))
            db.save_content(meta._db_id, "discord_embed", json.dumps(embed), "discord")

        if "blog" in formats:
            similar = db.find_similar(
                meta._embedding, limit=3, min_similarity=0.5,
                exclude_full_name=meta.full_name,
            )
            blog = format_blog_intro(meta, analysis, score, similar)
            (repo_dir / "blog_post.md").write_text(blog)
            db.save_content(meta._db_id, "blog_intro", blog, "blog")

        # Save raw analysis JSON
        (repo_dir / "analysis.json").write_text(json.dumps({
            "full_name": meta.full_name,
            "stars": meta.stars,
            "language": meta.language,
            "score": {
                "total": score.total,
                "tier": score.tier,
                "star_velocity": score.star_velocity,
                "freshness": score.freshness,
                "community_health": score.community_health,
                "adoption_signal": score.adoption_signal,
                "activity_momentum": score.activity_momentum,
                "documentation_quality": score.documentation_quality,
                "flags": score.flags,
            },
            "analysis": {
                "one_liner": analysis.one_liner,
                "problem_solved": analysis.problem_solved,
                "whats_novel": analysis.whats_novel,
                "who_should_care": analysis.who_should_care,
                "key_features": analysis.key_features,
                "categories": analysis.categories,
                "red_flags": analysis.red_flags,
            },
        }, indent=2))

    # Aggregated newsletter
    if "newsletter" in formats:
        newsletter_parts = []

        if synthesis:
            newsletter_parts.append(format_trend_synthesis_newsletter(synthesis, today))

        for (meta, score), analysis in zip(ranked, analyses):
            newsletter_parts.append(format_newsletter_entry(meta, analysis, score))

        full_newsletter = "\n\n".join(newsletter_parts)
        (output_dir / f"newsletter_{today}.md").write_text(full_newsletter)
        logger.info(f"Full newsletter saved to {output_dir}/newsletter_{today}.md")

    # Summary report
    summary_lines = [
        f"# discover-github Report — {today}",
        f"",
        f"**Strategy:** {strategy} | **Language:** {language or 'all'} | **Repos analyzed:** {len(ranked)}",
        f"**Duplicates skipped:** {skipped_dupes} | **Total in DB:** {db.get_repo_count()}",
        f"",
        f"## Top Repos by Breakout Score",
        f"",
    ]
    for i, (meta, score) in enumerate(ranked[:10]):
        analysis = analyses[i] if i < len(analyses) else None
        one_liner = analysis.one_liner if analysis else meta.description or ""
        summary_lines.append(
            f"{i+1}. **[{score.tier}] {meta.full_name}** — "
            f"Score: {score.total:.0f} | ⭐ {meta.stars:,} | "
            f"{one_liner[:100]}"
        )

    summary = "\n".join(summary_lines)
    (output_dir / f"report_{today}.md").write_text(summary)

    # Final summary
    logger.info(f"Done! Output saved to {output_dir}/")
    logger.info(f"Total repos in database: {db.get_repo_count()}")

    # Cleanup
    gh.close()
    db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent GitHub trending repository discovery engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategies:
  trending    New repos created in the last 24h with the most stars (default)
  hot         Repos created in the last 7 days with 50+ stars
  surging     Established repos (100+ stars) with recent push activity
  all         Combine all strategies and deduplicate

Examples:
  python discover.py --strategy hot --language rust --limit 15
  python discover.py --strategy all --skip-analysis --format linkedin
  python discover.py --strategy trending --output-dir ./daily-output
        """,
    )

    parser.add_argument(
        "--strategy", "-s",
        choices=["trending", "hot", "surging", "all"],
        default="trending",
        help="Discovery strategy (default: trending)",
    )
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="Filter by programming language (e.g., python, rust, go)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=25,
        help="Max repos to discover (default: 25)",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip LLM analysis (faster, metadata + scoring only)",
    )
    parser.add_argument(
        "--format", "-f",
        default="all",
        help="Content formats: twitter,linkedin,newsletter,discord,blog or 'all' (default: all)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="outputs",
        help="Output directory (default: ./outputs)",
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Config file path (default: config.yaml)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
