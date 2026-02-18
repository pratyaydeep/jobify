from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Portal:
    name: str
    url: str
    # Optional CSS selector to narrow down the jobs section on the page
    selector: str | None = None


@dataclass
class Profile:
    name: str
    title: str
    skills: list[str] = field(default_factory=list)
    experience_years: float = 0
    preferences: dict[str, str] = field(default_factory=dict)
    summary: str = ""


@dataclass
class Job:
    title: str
    company: str
    location: str
    url: str
    description: str
    source_portal: str
    min_experience_years: float | None = None
    max_experience_years: float | None = None  # None means no upper limit


@dataclass
class ScoredJob:
    job: Job
    score: float
    reasoning: str
