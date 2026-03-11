# discover-github — Intelligent GitHub Trending Discovery Engine

![](./img/repo_logo.png)

An intelligent content discovery engine that finds breakout GitHub repositories, scores them with a multi-signal algorithm, analyzes them with LLMs, and generates platform-ready social media content — all backed by Oracle AI Vector Search for semantic deduplication and trend tracking.

## What It Does

1. **Discovers** trending repos via GitHub API (3 strategies: trending, hot, surging)
2. **Enriches** with metadata: stars, forks, contributors, CI, README, topics, license
3. **Embeds** README content with sentence-transformers for semantic vector search
4. **Scores** each repo with a multi-signal breakout scoring engine (0-100, Tiers S/A/B/C/D)
5. **Deduplicates** against previously seen repos using Oracle AI Vector Search cosine similarity
6. **Analyzes** with structured LLM prompts via OCI GenAI ("Why This Matters" analysis)
7. **Generates** multi-platform content: Twitter threads, LinkedIn posts, newsletters, blog intros, Discord embeds
8. **Persists** everything to Oracle DB for historical trend tracking and analytics

## Breakout Scoring

Each repo gets a composite score from 6 weighted signals:

| Signal | Weight | What It Measures |
|--------|--------|-----------------|
| Star Velocity | 30% | Stars gained per day (log scale) |
| Freshness | 15% | Recency of last push |
| Community Health | 15% | Contributors, issues, CI pipeline |
| Adoption Signal | 15% | Fork-to-star ratio, watchers, homepage |
| Activity Momentum | 15% | Recent commits + age-adjusted star bonus |
| Documentation Quality | 10% | README length, license, topics, description |

Repos are tiered: **S** (80+) → **A** (65+) → **B** (50+) → **C** (35+) → **D** (<35)

## Requirements

- Python 3.10+
- Oracle Database 23ai+ (for AI Vector Search)
- [OCI SDK](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm) (for GenAI analysis)
- GitHub Personal Access Token (optional, increases rate limit from 60 to 5000 req/hr)

## Installation

<!-- one-command-install -->
> **One-command install** — clone, configure, and run in a single step:
>
> ```bash
> curl -fsSL https://raw.githubusercontent.com/jasperan/discover-github/main/install.sh | bash
> ```
>
> <details><summary>Advanced options</summary>
>
> Override install location:
> ```bash
> PROJECT_DIR=/opt/myapp curl -fsSL https://raw.githubusercontent.com/jasperan/discover-github/main/install.sh | bash
> ```
>
> Or install manually:
> ```bash
> git clone https://github.com/jasperan/discover-github.git
> cd discover-github
> ```
> </details>

```sh
conda create -n discover-github python=3.12
conda activate discover-github
pip install -r requirements.txt
```

## Configuration

Copy the example config and fill in your settings:

```bash
cp config.example.yaml config.yaml
```

```yaml
# GitHub token (optional, recommended for higher rate limits)
github_token: ""  # or set GITHUB_TOKEN env var

# Oracle Database connection
db_user: "discover_github"
db_password: "discover_github"
db_dsn: "localhost:1525/FREEPDB1"

# OCI GenAI (for LLM analysis)
compartment_id: "ocid1.compartment.oc1..your-id"
config_profile: "DEFAULT"
genai_endpoint: "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
genai_model: "cohere.command-r-plus"
```

The database schema is auto-created on first run.

## Usage

```bash
# Full pipeline with defaults (25 repos, all formats)
python discover.py

# Filter by language, use "hot recent" strategy
python discover.py --strategy hot --language python --limit 15

# Combine all strategies for maximum coverage
python discover.py --strategy all --limit 30

# Fast mode: skip LLM analysis, generate only LinkedIn posts
python discover.py --skip-analysis --format linkedin

# Generate everything with verbose logging
python discover.py --strategy all --limit 50 -v
```

### Discovery Strategies

| Strategy | What It Finds |
|----------|--------------|
| `trending` | New repos created in last 24h with the most stars (default) |
| `hot` | Repos created in last 7 days with 50+ stars |
| `surging` | Established repos (100+ stars) with recent push activity |
| `all` | Combines all 3 strategies, deduplicates |

### Output Formats

Generated content is saved per-repo in `outputs/<owner>__<repo>/`:

- `twitter_thread.txt` — Multi-tweet thread with hook, details, stats
- `linkedin_post.txt` — Professional post with hashtags and CTA
- `newsletter_entry.md` — Digest entry for email newsletters
- `discord_embed.json` — Rich embed for Discord/Slack webhooks
- `blog_post.md` — 800-word blog introduction
- `analysis.json` — Full scoring breakdown and LLM analysis

Daily aggregates:
- `outputs/newsletter_YYYY-MM-DD.md` — Full newsletter with trend synthesis
- `outputs/report_YYYY-MM-DD.md` — Summary ranking report

## Architecture

```
GitHub API ──→ GitHubClient ──→ RepoMetadata
                                    │
                            ┌───────┴───────┐
                            │               │
                    Embeddings          Scoring Engine
                    (MiniLM-L6)        (6 weighted signals)
                            │               │
                            ├───────┬───────┤
                            │       │       │
                    Oracle AI DB    LLM     Content
                    (Vector Search) Analyzer Formatter
                    (Dedup + Store) (OCI    (5 platforms)
                                    GenAI)
```

### Key Files

| File | Purpose |
|------|---------|
| `discover.py` | Main orchestrator and CLI entry point |
| `github_client.py` | GitHub API client with 3 discovery strategies |
| `db.py` | Oracle AI Vector Search persistence layer |
| `scoring.py` | Multi-signal breakout scoring engine |
| `analyzer.py` | OCI GenAI "Why This Matters" LLM analysis |
| `content_formatter.py` | Multi-platform content generation |
| `embeddings.py` | Sentence-transformers embedding generation |

### Legacy Pipeline

The original Scrapy-based pipeline (`trending_spider.py` → `readme_reader.py` → `main.py` → `summarize_llm.py`) is preserved for backward compatibility but is superseded by `discover.py`.

## LinkedIn Poster

### Pre-requisites

1. Create or use an existing developer application from the [LinkedIn Developer Portal](https://www.linkedin.com/developers/apps/)
2. Request access to the Sign In With LinkedIn API product
3. Generate a 3-legged access token using the Developer Portal [token generator tool](https://www.linkedin.com/developers/tools/oauth/token-generator), selecting the `r_liteprofile` scope

## License

This project is dual-licensed under the [Universal Permissive License (UPL) 1.0](https://oss.oracle.com/licenses/upl) and [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).
