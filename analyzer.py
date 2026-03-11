"""
'Why This Matters' LLM analyzer for discover-github.

Goes far beyond generic summarization — extracts structured insights:
- What problem does this solve and for whom
- What's genuinely novel about the approach
- How it compares to similar projects
- Project health assessment
- Content-ready social hooks

Uses OCI Generative AI with structured prompts.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import oci
import yaml

from github_client import RepoMetadata
from scoring import ScoreBreakdown

logger = logging.getLogger(__name__)


@dataclass
class RepoAnalysis:
    """Structured analysis of a repository."""
    repo_full_name: str
    one_liner: str = ""           # Single-sentence hook
    problem_solved: str = ""      # What pain point does this address
    whats_novel: str = ""         # How is the approach different/better
    who_should_care: str = ""     # Target audience
    key_features: list[str] = field(default_factory=list)
    potential_impact: str = ""    # Why this matters for the ecosystem
    competitive_landscape: str = ""  # How it compares to alternatives
    red_flags: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    social_hooks: list[str] = field(default_factory=list)  # Attention-grabbing angles
    raw_summary: str = ""         # Fallback plain summary


ANALYSIS_PROMPT = """You are an expert tech analyst writing for developers and tech leaders.

Analyze this GitHub repository and provide a structured JSON response.

## Repository Info
- **Name**: {full_name}
- **Description**: {description}
- **Language**: {language}
- **Stars**: {stars:,} | **Forks**: {forks:,} | **Contributors**: {contributors}
- **Topics**: {topics}
- **License**: {license}
- **Age**: {age_days} days | **Last push**: {pushed_at}
- **Breakout Score**: {breakout_score}/100 (Tier {tier})
- **Score Flags**: {flags}

## README Content (truncated)
{readme}

## Instructions
Respond with ONLY valid JSON matching this schema:
{{
    "one_liner": "A single compelling sentence that makes someone stop scrolling",
    "problem_solved": "What specific pain point this addresses (2-3 sentences)",
    "whats_novel": "What makes the approach different from existing solutions (2-3 sentences)",
    "who_should_care": "Specific audience: e.g. 'ML engineers working with LLMs', not just 'developers'",
    "key_features": ["feature1", "feature2", "feature3"],
    "potential_impact": "Why this matters for the broader tech ecosystem (1-2 sentences)",
    "competitive_landscape": "Name 1-3 alternatives and how this compares",
    "red_flags": ["any concerns about sustainability, quality, or adoption"],
    "categories": ["primary-category", "secondary-category"],
    "social_hooks": [
        "Hook 1: attention-grabbing angle for social media",
        "Hook 2: different angle or surprising fact"
    ]
}}

Categories should be from: AI/ML, DevOps, Security, Web, Mobile, Systems, Data, Cloud,
Developer-Tools, Infrastructure, Blockchain, IoT, Gaming, Finance, Education, Other.

Be specific and insightful — avoid generic platitudes. If the README is thin, say so in red_flags.
"""


SYNTHESIS_PROMPT = """You are a tech trend analyst. Given these {count} trending GitHub repositories
discovered today, identify cross-cutting themes and patterns.

## Repositories
{repo_summaries}

## Instructions
Respond with ONLY valid JSON:
{{
    "meta_trends": [
        {{
            "trend": "Short trend title",
            "description": "2-3 sentence explanation of the pattern",
            "repos": ["repo1/name", "repo2/name"],
            "significance": "Why this trend matters"
        }}
    ],
    "overall_narrative": "A 2-3 sentence synthesis of what today's trending repos tell us about where tech is heading",
    "surprise_factor": "The most unexpected or counterintuitive finding"
}}

Look for patterns like: language migrations, new paradigm adoption, infrastructure shifts,
AI application patterns, security focus areas, developer tool consolidation.
"""


class RepoAnalyzer:
    """LLM-powered repository analysis using OCI Generative AI."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        self.compartment_id = config_data["compartment_id"]
        oci_profile = config_data.get("config_profile", "DEFAULT")
        oci_config = oci.config.from_file("~/.oci/config", oci_profile)

        endpoint = config_data.get(
            "genai_endpoint",
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        )

        self.model_id = config_data.get("genai_model", "cohere.command-r-plus")

        self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=oci_config,
            service_endpoint=endpoint,
            retry_strategy=oci.retry.NoneRetryStrategy(),
            timeout=(10, 240),
        )

    def analyze_repo(
        self,
        meta: RepoMetadata,
        score: ScoreBreakdown,
        similar_repos: list[dict] | None = None,
    ) -> RepoAnalysis:
        """
        Generate a structured 'Why This Matters' analysis for a repo.
        """
        readme_truncated = ""
        if meta.readme_content:
            readme_truncated = meta.readme_content[:8000]

        prompt = ANALYSIS_PROMPT.format(
            full_name=meta.full_name,
            description=meta.description or "No description",
            language=meta.language or "Unknown",
            stars=meta.stars,
            forks=meta.forks,
            contributors=meta.contributor_count,
            topics=", ".join(meta.topics) if meta.topics else "None",
            license=meta.license_name or "None",
            age_days=meta.age_days(),
            pushed_at=meta.pushed_at.strftime("%Y-%m-%d") if meta.pushed_at else "Unknown",
            breakout_score=score.total,
            tier=score.tier,
            flags=", ".join(score.flags) if score.flags else "None",
            readme=readme_truncated,
        )

        # Add similar repos context if available
        if similar_repos:
            similar_text = "\n## Similar Repos Already in Our Database\n"
            for s in similar_repos[:3]:
                similar_text += f"- {s['full_name']} ({s['stars']} stars, {s['similarity']:.0%} similar): {s['description']}\n"
            prompt += similar_text
            prompt += "\nIncorporate comparisons to these in your competitive_landscape.\n"

        response_text = self._call_llm(prompt)
        return self._parse_analysis(meta.full_name, response_text)

    def synthesize_trends(
        self,
        analyses: list[RepoAnalysis],
        scores: list[ScoreBreakdown],
    ) -> dict:
        """
        Generate cross-repo trend synthesis — the most differentiated feature.
        Identifies meta-patterns across today's discovered repos.
        """
        repo_summaries = ""
        for analysis, score in zip(analyses, scores):
            repo_summaries += (
                f"- **{analysis.repo_full_name}** (Score: {score.total}, Tier {score.tier}): "
                f"{analysis.one_liner}\n"
                f"  Categories: {', '.join(analysis.categories)}\n"
                f"  Novel: {analysis.whats_novel}\n\n"
            )

        prompt = SYNTHESIS_PROMPT.format(
            count=len(analyses),
            repo_summaries=repo_summaries,
        )

        response_text = self._call_llm(prompt)
        try:
            return json.loads(self._extract_json(response_text))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse trend synthesis JSON")
            return {
                "meta_trends": [],
                "overall_narrative": response_text[:500],
                "surprise_factor": "",
            }

    def _call_llm(self, prompt: str) -> str:
        """Call OCI GenAI with a chat prompt."""
        chat_detail = oci.generative_ai_inference.models.ChatDetails()
        chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
            model_id=self.model_id,
        )
        chat_detail.compartment_id = self.compartment_id
        chat_detail.chat_request = oci.generative_ai_inference.models.CohereChatRequest(
            message=prompt,
            max_tokens=2048,
            temperature=0.3,
            is_stream=False,
        )

        try:
            response = self.client.chat(chat_detail)
            return response.data.chat_response.text
        except oci.exceptions.ServiceError as e:
            logger.error(f"OCI GenAI error: {e}")
            # Fallback: try the summarize endpoint with truncated input
            return self._fallback_summarize(prompt[:4000])

    def _fallback_summarize(self, text: str) -> str:
        """Fallback to the original summarize API if chat fails."""
        try:
            summarize_detail = oci.generative_ai_inference.models.SummarizeTextDetails()
            summarize_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
                model_id="cohere.command",
            )
            summarize_detail.compartment_id = self.compartment_id
            summarize_detail.input = text
            summarize_detail.additional_command = (
                "Generate a teaser summary. Share an interesting insight to captivate attention."
            )
            summarize_detail.length = "LONG"
            summarize_detail.temperature = 0.25

            response = self.client.summarize_text(summarize_detail)
            return response.data.summary
        except Exception as e:
            logger.error(f"Fallback summarize also failed: {e}")
            return ""

    def _parse_analysis(self, full_name: str, response_text: str) -> RepoAnalysis:
        """Parse LLM JSON response into a RepoAnalysis dataclass."""
        analysis = RepoAnalysis(repo_full_name=full_name)
        analysis.raw_summary = response_text

        try:
            data = json.loads(self._extract_json(response_text))
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Failed to parse JSON for {full_name}, using raw text")
            analysis.one_liner = response_text[:200]
            return analysis

        analysis.one_liner = data.get("one_liner", "")
        analysis.problem_solved = data.get("problem_solved", "")
        analysis.whats_novel = data.get("whats_novel", "")
        analysis.who_should_care = data.get("who_should_care", "")
        analysis.key_features = data.get("key_features", [])
        analysis.potential_impact = data.get("potential_impact", "")
        analysis.competitive_landscape = data.get("competitive_landscape", "")
        analysis.red_flags = data.get("red_flags", [])
        analysis.categories = data.get("categories", [])
        analysis.social_hooks = data.get("social_hooks", [])

        return analysis

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from LLM response that may contain markdown fences."""
        # Try to find JSON in code fences first
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)

        # Try to find raw JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)

        return text
