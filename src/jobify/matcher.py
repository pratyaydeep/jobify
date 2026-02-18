from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import yaml

from jobify.llm import filter_job_links, parse_job_page, score_job
from jobify.models import Job, Portal, Profile, ScoredJob
from jobify.scraper import scrape_job_pages, scrape_portal

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"


def load_profile(path: Path | None = None) -> Profile:
    path = path or CONFIG_DIR / "profile.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return Profile(
        name=data.get("name", ""),
        title=data.get("title", ""),
        skills=data.get("skills", []),
        experience_years=data.get("experience_years", 0),
        preferences=data.get("preferences", {}),
        summary=data.get("summary", ""),
    )


def load_portals(path: Path | None = None) -> list[Portal]:
    path = path or CONFIG_DIR / "portals.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return [
        Portal(
            name=p["name"],
            url=p["url"],
            selector=p.get("selector"),
        )
        for p in data.get("portals", [])
    ]


async def run_pipeline(
    profile: Profile,
    portals: list[Portal],
    on_status: callable = None,
) -> list[ScoredJob]:
    """Full pipeline: scrape -> filter links -> visit jobs -> parse -> score -> rank."""

    def status(msg: str) -> None:
        if on_status:
            on_status(msg)

    all_jobs: list[Job] = []

    for portal in portals:
        status(f"Scraping {portal.name}...")

        # 1. Scrape the career page: scroll, paginate, extract links
        try:
            links, _page_text = await scrape_portal(portal)
        except Exception as exc:
            status(f"  Failed to scrape {portal.name}: {exc}")
            continue

        status(f"  Found {len(links)} links on {portal.name}")

        # 2. Ask the LLM which links are job postings
        status(f"  Identifying job links on {portal.name}...")
        link_dicts = [{"text": l.text, "href": l.href} for l in links]
        job_urls = await filter_job_links(link_dicts, portal.name)
        status(f"  Identified {len(job_urls)} job links on {portal.name}")

        if not job_urls:
            continue

        # Cap to avoid spending too long on a single portal
        MAX_JOBS_PER_PORTAL = 25
        if len(job_urls) > MAX_JOBS_PER_PORTAL:
            status(f"  Capping to {MAX_JOBS_PER_PORTAL} (from {len(job_urls)}) job links")
            job_urls = job_urls[:MAX_JOBS_PER_PORTAL]

        # 3. Visit each job page and scrape its content
        status(f"  Scraping {len(job_urls)} job pages from {portal.name}...")
        job_page_texts = await scrape_job_pages(job_urls)

        # 4. Parse each job page with the LLM
        status(f"  Parsing job details from {portal.name}...")
        for url, text in job_page_texts.items():
            if not text:
                continue
            job = await parse_job_page(text, portal.name, url)
            if job:
                all_jobs.append(job)

        status(f"  Extracted {len(all_jobs)} jobs so far")

    # Save raw jobs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "jobs.json", "w") as f:
        json.dump([asdict(j) for j in all_jobs], f, indent=2)

    # Filter: only keep jobs where the user's experience falls in the required range
    # Range logic:
    #   "3-7 years"  -> min=3, max=7  -> eligible if 3 <= user_exp <= 7
    #   "2+ years"   -> min=2, max=None -> eligible if user_exp >= 2
    #   no range      -> min=None, max=None -> always eligible
    before_filter = len(all_jobs)
    user_exp = profile.experience_years

    def is_eligible(job: Job) -> bool:
        if job.min_experience_years is not None and user_exp < job.min_experience_years:
            return False
        if job.max_experience_years is not None and user_exp > job.max_experience_years:
            return False
        return True

    eligible_jobs = [j for j in all_jobs if is_eligible(j)]
    filtered_out = before_filter - len(eligible_jobs)
    if filtered_out:
        status(f"Filtered out {filtered_out} jobs outside your experience range ({user_exp} years)")
    all_jobs = eligible_jobs

    status(f"Eligible jobs: {len(all_jobs)}. Scoring against your profile...")

    # 5. Score each job against the profile
    scored: list[ScoredJob] = []
    for i, job in enumerate(all_jobs, 1):
        status(f"  Scoring job {i}/{len(all_jobs)}: {job.title}")
        score, reasoning = await score_job(job, profile)
        scored.append(ScoredJob(job=job, score=score, reasoning=reasoning))

    # 6. Rank by score
    scored.sort(key=lambda s: s.score, reverse=True)

    # 7. Save suggestions
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "suggestions.json", "w") as f:
        json.dump([asdict(s) for s in scored], f, indent=2)

    return scored
