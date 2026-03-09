from __future__ import annotations

import json
import logging
import os

import httpx
from dotenv import load_dotenv

from jobify.models import Job, Profile

load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_BASE = os.getenv("JOBIFY_OLLAMA_BASE", "http://localhost:11434")
MODEL = os.getenv("JOBIFY_MODEL", "gemma3n")


async def _generate(prompt: str, model: str = MODEL) -> str:
    """Call Ollama's /api/generate endpoint and return the response text."""
    logger.debug("LLM request to %s model=%s prompt_len=%d", OLLAMA_BASE, model, len(prompt))
    async with httpx.AsyncClient(timeout=600.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        response = resp.json()["response"]
        logger.debug("LLM response len=%d", len(response))
        return response


def _parse_json_array(text: str) -> list[dict]:
    """Best-effort parse of a JSON array from LLM output."""
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []


def _parse_json_object(text: str) -> dict:
    """Best-effort parse of a JSON object from LLM output."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Step 1: Identify which links on a careers page are actual job postings
# ---------------------------------------------------------------------------

def _build_filter_links_prompt(
    links: list[dict[str, str]], portal_name: str
) -> str:
    # Number each link so the LLM returns indices instead of URLs
    links_text = "\n".join(
        f'{i}: text="{l["text"][:100]}"  href="{l["href"]}"'
        for i, l in enumerate(links[:200])
    )
    return f"""You are analyzing links from the career page of {portal_name}.
Identify which links point to individual job postings (not blog posts, not general info pages, not navigation links).

Here are the links found on the page (numbered):
{links_text}

Return a JSON array of the NUMBERS (indices) of links that are job posting links. Example:
[3, 7, 12, 15]

IMPORTANT: Return ONLY the JSON array of numbers, no other text."""


async def filter_job_links(
    links: list[dict[str, str]], portal_name: str
) -> list[str]:
    """Use the LLM to identify which page links are job postings.

    Returns the actual hrefs from the scraped links (never LLM-generated URLs).
    """
    if not links:
        return []
    capped = links[:200]
    logger.info("Filtering %d links for portal %s", len(capped), portal_name)
    prompt = _build_filter_links_prompt(capped, portal_name)
    raw = await _generate(prompt)
    indices = _parse_json_array(raw)
    # Map indices back to real hrefs
    result: list[str] = []
    for idx in indices:
        if isinstance(idx, int) and 0 <= idx < len(capped):
            result.append(capped[idx]["href"])
    return result


# ---------------------------------------------------------------------------
# Step 2: Parse a single job posting page into structured data
# ---------------------------------------------------------------------------

def _build_parse_job_prompt(page_text: str, portal_name: str) -> str:
    return f"""You are extracting job details from a single job posting page.

Company: {portal_name}

Extract the following and return as a JSON object:
- "title": the job title
- "company": company name
- "location": job location(s), comma-separated (use "Not specified" if unclear)
- "description": a 2-3 sentence summary of the role and its key responsibilities
- "min_experience_years": the minimum years of experience required as a number (use null if not mentioned). Look for patterns like "X+ years", "X-Y years", "at least X years" in the job posting. Also infer from seniority level in the title (e.g. "Senior" typically means 5+, "Staff" means 8+, "Lead" means 7+, "Principal" means 10+, "Director" means 12+).
- "max_experience_years": the maximum years of experience as a number. For "X-Y years" use Y. For "X+ years" or open-ended ranges use null (meaning no upper limit). Use null if not mentioned.

IMPORTANT: Return ONLY the JSON object, no other text.

--- PAGE TEXT START ---
{page_text[:6000]}
--- PAGE TEXT END ---"""


async def parse_job_page(
    page_text: str, portal_name: str, job_url: str
) -> Job | None:
    """Use the LLM to parse a job posting page into a Job object.

    The job_url is always the real scraped URL — never LLM-generated.
    """
    if not page_text.strip():
        return None
    logger.info("Parsing job page from %s: %s", portal_name, job_url)
    prompt = _build_parse_job_prompt(page_text, portal_name)
    raw = await _generate(prompt)
    data = _parse_json_object(raw)
    if not data:
        logger.warning("Failed to parse JSON from LLM response for %s", job_url)
        return None
    min_exp = data.get("min_experience_years")
    if min_exp is not None:
        try:
            min_exp = float(min_exp)
        except (ValueError, TypeError):
            min_exp = None

    max_exp = data.get("max_experience_years")
    if max_exp is not None:
        try:
            max_exp = float(max_exp)
        except (ValueError, TypeError):
            max_exp = None

    return Job(
        title=data.get("title", "Unknown"),
        company=data.get("company", portal_name),
        location=data.get("location", "Not specified"),
        url=job_url,  # always use the real scraped URL
        description=data.get("description", ""),
        source_portal=portal_name,
        min_experience_years=min_exp,
        max_experience_years=max_exp,
    )


# ---------------------------------------------------------------------------
# Step 3: Score a job against the user's profile
# ---------------------------------------------------------------------------

def _build_scoring_prompt(job: Job, profile: Profile) -> str:
    return f"""You are a job matching assistant. Score how well this job matches the candidate's profile.

Return a JSON object with:
- "score": integer from 1 to 10 (10 = perfect match)
- "reasoning": 1-2 sentences explaining the score

JOB:
- Title: {job.title}
- Company: {job.company}
- Location: {job.location}
- Description: {job.description}

CANDIDATE:
- Name: {profile.name}
- Current title: {profile.title}
- Skills: {', '.join(profile.skills)}
- Experience: {profile.experience_years} years
- Preferences: {json.dumps(profile.preferences)}
- Summary: {profile.summary}

IMPORTANT: Return ONLY the JSON object, no other text."""


async def score_job(job: Job, profile: Profile) -> tuple[float, str]:
    """Use the LLM to score how well a job matches the profile."""
    logger.info("Scoring job: %s at %s", job.title, job.company)
    prompt = _build_scoring_prompt(job, profile)
    raw = await _generate(prompt)
    result = _parse_json_object(raw)
    score = float(result.get("score", 0))
    reasoning = result.get("reasoning", "No reasoning provided.")
    return score, reasoning
