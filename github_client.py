"""
GitHub API client for discovering trending repositories.

Replaces the fragile Scrapy XPath spider with the GitHub REST API,
providing rich metadata (stars, forks, contributors, languages, topics)
that powers the breakout scoring engine and intelligent analysis.
"""

import os
import logging
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional

from github import Github, Auth
from github.GithubException import RateLimitExceededException, GithubException

logger = logging.getLogger(__name__)


@dataclass
class RepoMetadata:
    """Rich metadata for a discovered repository."""
    full_name: str                    # e.g. "openai/whisper"
    owner: str
    name: str
    description: Optional[str]
    url: str
    homepage: Optional[str]
    language: Optional[str]
    topics: list[str] = field(default_factory=list)

    # Counts
    stars: int = 0
    forks: int = 0
    open_issues: int = 0
    watchers: int = 0
    subscribers: int = 0

    # Activity signals
    created_at: Optional[datetime] = None
    pushed_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    default_branch: str = "main"
    license_name: Optional[str] = None
    is_fork: bool = False
    is_archived: bool = False
    size_kb: int = 0

    # Computed during enrichment
    contributor_count: int = 0
    recent_commit_count: int = 0   # commits in last 7 days
    star_history_24h: int = 0      # approximate via stargazers page
    has_ci: bool = False           # .github/workflows exists
    readme_content: Optional[str] = None

    def age_days(self) -> int:
        if self.created_at:
            return (datetime.now(timezone.utc) - self.created_at).days
        return 0


class GitHubClient:
    """Client for discovering and enriching GitHub repository metadata."""

    def __init__(self, token: Optional[str] = None):
        """
        Initialize with optional GitHub Personal Access Token.
        Without a token, rate limit is 60 req/hr. With token: 5000 req/hr.
        """
        if token:
            self.gh = Github(auth=Auth.Token(token))
        else:
            # Check environment variable
            env_token = os.environ.get("GITHUB_TOKEN")
            if env_token:
                self.gh = Github(auth=Auth.Token(env_token))
            else:
                logger.warning("No GitHub token provided. Rate limit: 60 req/hr.")
                self.gh = Github()

    def discover_trending(
        self,
        language: Optional[str] = None,
        since_days: int = 1,
        limit: int = 25,
        min_stars: int = 10,
    ) -> list[RepoMetadata]:
        """
        Discover trending repositories using GitHub Search API.

        Uses 'created:>DATE sort:stars' to find repos gaining traction.
        This is more reliable than scraping the trending page and gives
        us rich metadata for free.
        """
        since_date = (datetime.now(timezone.utc) - timedelta(days=since_days)).strftime("%Y-%m-%d")

        query_parts = [f"created:>{since_date}", f"stars:>={min_stars}"]
        if language:
            query_parts.append(f"language:{language}")

        query = " ".join(query_parts)
        logger.info(f"Searching GitHub: {query}")

        try:
            results = self.gh.search_repositories(
                query=query,
                sort="stars",
                order="desc",
            )
        except RateLimitExceededException:
            logger.error("GitHub API rate limit exceeded. Provide a GITHUB_TOKEN.")
            return []
        except GithubException as e:
            logger.error(f"GitHub API error: {e}")
            return []

        repos = []
        for i, repo in enumerate(results):
            if i >= limit:
                break
            repos.append(self._extract_metadata(repo))

        logger.info(f"Discovered {len(repos)} repositories")
        return repos

    def discover_hot_recent(
        self,
        days_back: int = 7,
        limit: int = 25,
        min_stars: int = 50,
        language: Optional[str] = None,
    ) -> list[RepoMetadata]:
        """
        Find repos created in the last N days with the most stars.
        These are the true "breakout" repos — new and already popular.
        """
        since_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

        query_parts = [f"created:>{since_date}", f"stars:>={min_stars}"]
        if language:
            query_parts.append(f"language:{language}")

        query = " ".join(query_parts)

        try:
            results = self.gh.search_repositories(query=query, sort="stars", order="desc")
        except GithubException as e:
            logger.error(f"GitHub API error: {e}")
            return []

        repos = []
        for i, repo in enumerate(results):
            if i >= limit:
                break
            repos.append(self._extract_metadata(repo))

        return repos

    def discover_surging(
        self,
        days_back: int = 30,
        limit: int = 25,
        min_stars: int = 100,
        language: Optional[str] = None,
    ) -> list[RepoMetadata]:
        """
        Find established repos that were recently updated and are surging.
        Sorted by 'updated' to catch repos with sudden activity spikes.
        """
        since_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

        query_parts = [f"pushed:>{since_date}", f"stars:>={min_stars}"]
        if language:
            query_parts.append(f"language:{language}")

        query = " ".join(query_parts)

        try:
            results = self.gh.search_repositories(query=query, sort="updated", order="desc")
        except GithubException as e:
            logger.error(f"GitHub API error: {e}")
            return []

        repos = []
        for i, repo in enumerate(results):
            if i >= limit:
                break
            repos.append(self._extract_metadata(repo))

        return repos

    def enrich_repo(self, meta: RepoMetadata) -> RepoMetadata:
        """
        Enrich a RepoMetadata with additional API calls:
        - Contributor count
        - Recent commit count (last 7 days)
        - CI detection
        - README content (using correct default branch)
        """
        try:
            repo = self.gh.get_repo(meta.full_name)

            # Contributor count (paginated, cap at 500 to save API calls)
            try:
                contributors = repo.get_contributors()
                meta.contributor_count = min(contributors.totalCount, 500)
            except GithubException:
                meta.contributor_count = 0

            # Recent commits (last 7 days)
            since = datetime.now(timezone.utc) - timedelta(days=7)
            try:
                commits = repo.get_commits(since=since)
                meta.recent_commit_count = min(commits.totalCount, 1000)
            except GithubException:
                meta.recent_commit_count = 0

            # CI detection
            try:
                repo.get_contents(".github/workflows")
                meta.has_ci = True
            except GithubException:
                meta.has_ci = False

            # README content (uses correct default branch automatically)
            try:
                readme = repo.get_readme()
                meta.readme_content = readme.decoded_content.decode("utf-8", errors="replace")
            except GithubException:
                meta.readme_content = None

        except RateLimitExceededException:
            logger.warning(f"Rate limit hit while enriching {meta.full_name}, retrying with backoff...")
            for attempt in range(3):
                wait = 2 ** (attempt + 1)  # 2, 4, 8 seconds
                logger.info(f"  Backoff attempt {attempt + 1}/3: waiting {wait}s...")
                time.sleep(wait)
                try:
                    return self.enrich_repo(meta)
                except RateLimitExceededException:
                    if attempt == 2:
                        logger.error(f"Rate limit still exceeded after 3 retries for {meta.full_name}")
                except GithubException as retry_err:
                    logger.warning(f"Retry failed for {meta.full_name}: {retry_err}")
                    break
        except GithubException as e:
            logger.warning(f"Error enriching {meta.full_name}: {e}")

        return meta

    def _extract_metadata(self, repo) -> RepoMetadata:
        """Extract RepoMetadata from a PyGithub Repository object."""
        return RepoMetadata(
            full_name=repo.full_name,
            owner=repo.owner.login,
            name=repo.name,
            description=repo.description,
            url=repo.html_url,
            homepage=repo.homepage,
            language=repo.language,
            topics=repo.topics or [],
            stars=repo.stargazers_count,
            forks=repo.forks_count,
            open_issues=repo.open_issues_count,
            watchers=repo.watchers_count,
            subscribers=repo.subscribers_count,
            created_at=repo.created_at.replace(tzinfo=timezone.utc) if repo.created_at else None,
            pushed_at=repo.pushed_at.replace(tzinfo=timezone.utc) if repo.pushed_at else None,
            updated_at=repo.updated_at.replace(tzinfo=timezone.utc) if repo.updated_at else None,
            default_branch=repo.default_branch,
            license_name=repo.license.name if repo.license else None,
            is_fork=repo.fork,
            is_archived=repo.archived,
            size_kb=repo.size,
        )

    def close(self):
        self.gh.close()
