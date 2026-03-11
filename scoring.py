"""
Breakout scoring engine for discovered GitHub repositories.

Computes a composite score from multiple signals to identify repos
that are genuinely breaking out — not just momentarily popular.
Finds gems BEFORE they peak on the trending page.
"""

import math
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from github_client import RepoMetadata

logger = logging.getLogger(__name__)


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of a repo's breakout score."""
    total: float
    star_velocity: float
    freshness: float
    community_health: float
    adoption_signal: float
    activity_momentum: float
    documentation_quality: float
    tier: str  # "S", "A", "B", "C", "D"
    flags: list[str]  # e.g. ["no-license", "single-contributor", "archived"]


# Weights for each signal (must sum to 1.0)
WEIGHTS = {
    "star_velocity": 0.30,
    "freshness": 0.15,
    "community_health": 0.15,
    "adoption_signal": 0.15,
    "activity_momentum": 0.15,
    "documentation_quality": 0.10,
}


def compute_breakout_score(
    meta: RepoMetadata,
    star_velocity_per_day: float | None = None,
) -> ScoreBreakdown:
    """
    Compute a 0-100 breakout score from multiple signals.

    Args:
        meta: Rich repository metadata from GitHub API.
        star_velocity_per_day: Stars gained per day (from DB snapshots).
                               If None, estimated from repo age.
    """
    flags = []

    # --- 1. Star Velocity (0-100) ---
    # How fast is this repo gaining stars?
    if star_velocity_per_day is not None:
        sv = star_velocity_per_day
    else:
        age = max(meta.age_days(), 1)
        sv = meta.stars / age

    # Logarithmic scale: 1 star/day = ~30, 10/day = ~60, 100/day = ~85, 1000/day = ~100
    star_velocity_score = min(100, max(0, 30 * math.log10(max(sv, 0.1) + 1)))

    # --- 2. Freshness (0-100) ---
    # How recent is the last push? Newer = better.
    if meta.pushed_at:
        days_since_push = (datetime.now(timezone.utc) - meta.pushed_at).total_seconds() / 86400
    else:
        days_since_push = 365

    if days_since_push <= 1:
        freshness_score = 100
    elif days_since_push <= 7:
        freshness_score = 90 - (days_since_push * 2)
    elif days_since_push <= 30:
        freshness_score = 75 - (days_since_push - 7)
    elif days_since_push <= 90:
        freshness_score = max(20, 50 - (days_since_push - 30))
    else:
        freshness_score = max(0, 20 - (days_since_push - 90) * 0.1)

    # --- 3. Community Health (0-100) ---
    # Multi-contributor projects with active issues are healthier.
    contributor_score = min(40, meta.contributor_count * 4)  # 10+ contributors = max

    issue_engagement = 0
    if meta.open_issues > 0:
        # Some open issues = healthy community. Too many = maybe abandoned.
        if meta.open_issues <= 50:
            issue_engagement = 30
        elif meta.open_issues <= 200:
            issue_engagement = 20
        else:
            issue_engagement = 10

    ci_score = 30 if meta.has_ci else 0

    community_health_score = min(100, contributor_score + issue_engagement + ci_score)

    if meta.contributor_count <= 1:
        flags.append("single-contributor")

    # --- 4. Adoption Signal (0-100) ---
    # Fork-to-star ratio indicates real usage, not just interest.
    if meta.stars > 0:
        fork_ratio = meta.forks / meta.stars
    else:
        fork_ratio = 0

    # Healthy fork ratio is 0.1-0.3. Below = mostly spectators. Above = utility library.
    if 0.05 <= fork_ratio <= 0.5:
        fork_score = 50
    elif fork_ratio > 0.5:
        fork_score = 40  # Very high fork ratio = useful but maybe not novel
    else:
        fork_score = 20  # Low fork ratio = people star but don't use

    # Watchers/subscribers indicate sustained interest
    watcher_score = min(30, meta.subscribers * 2)

    # Homepage = project is serious enough to have a website
    homepage_score = 20 if meta.homepage else 0

    adoption_signal_score = min(100, fork_score + watcher_score + homepage_score)

    # --- 5. Activity Momentum (0-100) ---
    # Recent commits indicate active development.
    if meta.recent_commit_count >= 50:
        commit_score = 60
    elif meta.recent_commit_count >= 20:
        commit_score = 50
    elif meta.recent_commit_count >= 10:
        commit_score = 40
    elif meta.recent_commit_count >= 5:
        commit_score = 30
    elif meta.recent_commit_count >= 1:
        commit_score = 20
    else:
        commit_score = 0

    # Young repos with high stars get a momentum bonus
    age = max(meta.age_days(), 1)
    if age <= 7 and meta.stars >= 100:
        momentum_bonus = 40
    elif age <= 30 and meta.stars >= 500:
        momentum_bonus = 30
    elif age <= 90 and meta.stars >= 1000:
        momentum_bonus = 20
    else:
        momentum_bonus = 0

    activity_momentum_score = min(100, commit_score + momentum_bonus)

    # --- 6. Documentation Quality (0-100) ---
    # Good README, license, and topics indicate a serious project.
    doc_score = 0

    if meta.readme_content:
        readme_len = len(meta.readme_content)
        if readme_len >= 5000:
            doc_score += 30
        elif readme_len >= 1000:
            doc_score += 20
        elif readme_len >= 200:
            doc_score += 10
    else:
        flags.append("no-readme")

    if meta.license_name:
        doc_score += 25
    else:
        flags.append("no-license")

    if meta.topics and len(meta.topics) >= 3:
        doc_score += 20
    elif meta.topics:
        doc_score += 10

    if meta.description and len(meta.description) >= 20:
        doc_score += 25
    elif meta.description:
        doc_score += 10

    documentation_quality_score = min(100, doc_score)

    # --- Penalties ---
    if meta.is_archived:
        flags.append("archived")
        star_velocity_score *= 0.1
        activity_momentum_score = 0

    if meta.is_fork:
        flags.append("is-fork")
        star_velocity_score *= 0.5

    # --- Weighted Total ---
    total = (
        WEIGHTS["star_velocity"] * star_velocity_score
        + WEIGHTS["freshness"] * freshness_score
        + WEIGHTS["community_health"] * community_health_score
        + WEIGHTS["adoption_signal"] * adoption_signal_score
        + WEIGHTS["activity_momentum"] * activity_momentum_score
        + WEIGHTS["documentation_quality"] * documentation_quality_score
    )

    # Determine tier
    if total >= 80:
        tier = "S"
    elif total >= 65:
        tier = "A"
    elif total >= 50:
        tier = "B"
    elif total >= 35:
        tier = "C"
    else:
        tier = "D"

    return ScoreBreakdown(
        total=round(total, 2),
        star_velocity=round(star_velocity_score, 2),
        freshness=round(freshness_score, 2),
        community_health=round(community_health_score, 2),
        adoption_signal=round(adoption_signal_score, 2),
        activity_momentum=round(activity_momentum_score, 2),
        documentation_quality=round(documentation_quality_score, 2),
        tier=tier,
        flags=flags,
    )


def rank_repos(
    repos: list[RepoMetadata],
    star_velocities: dict[str, float] | None = None,
) -> list[tuple[RepoMetadata, ScoreBreakdown]]:
    """
    Score and rank a list of repos by breakout potential.
    Returns sorted list of (metadata, score_breakdown) tuples.
    """
    if star_velocities is None:
        star_velocities = {}

    scored = []
    for meta in repos:
        velocity = star_velocities.get(meta.full_name)
        score = compute_breakout_score(meta, velocity)
        scored.append((meta, score))

    scored.sort(key=lambda x: x[1].total, reverse=True)
    return scored
