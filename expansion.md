# Expansion Plan

Ideas for scaling, hardening, and growing Jobify into a product.

---

## Architectural Improvements

### 1. Parallel & Concurrent Processing

- **Concurrent LLM scoring** — Jobs are currently scored sequentially (N jobs = N serial LLM calls). Use `asyncio.gather()` or a semaphore-bounded task pool to score multiple jobs in parallel. With a concurrency of 5, a 50-job run drops from ~25 min to ~5 min.
- **Parallel portal scraping** — Scrape multiple career portals simultaneously instead of one at a time. Each portal gets its own browser context.
- **Browser connection pooling** — Reuse a single Chromium instance across portals instead of launching/closing a new one per portal. Use `browser.new_context()` for isolation without the process overhead.

### 2. Caching & Incremental Runs

- **Job fingerprinting** — Hash each job by (URL, title, company) and store in a local SQLite database. On subsequent runs, skip jobs already seen and only process new postings.
- **Portal content hashing** — Store a hash of each portal's page content. If the page hasn't changed since the last run, skip the full scrape entirely.
- **LLM response caching** — Cache LLM responses keyed by prompt hash. Useful when re-running after a crash — already-parsed jobs don't need to hit the LLM again.

### 3. Database Layer

Replace flat JSON files with a proper database (SQLite for single-user, PostgreSQL for multi-user):

- `jobs` table — All extracted jobs with timestamps, dedup keys, and status
- `scored_jobs` table — Scores tied to a specific profile version
- `run_history` table — Track each pipeline run (portals scraped, jobs found, duration)
- `portals` table — Move portal config to DB for dynamic management
- Enables querying, filtering, search, and historical trends

### 4. Job Deduplication

- **Cross-portal dedup** — The same role posted on a company's own site and an aggregator should be merged. Fuzzy-match on (company + title + location) using string similarity.
- **Cross-run dedup** — Don't re-score jobs from previous runs. Show "new since last run" vs "previously seen".
- **URL normalization** — Strip tracking params, fragments, and session IDs before comparing URLs.

### 5. LLM Flexibility

- **Environment-based config** — Move `OLLAMA_BASE` and `MODEL` to environment variables or a settings file instead of hardcoding.
- **Cloud LLM fallback** — If Ollama is unavailable, fall back to an API provider (Claude, OpenAI, etc.). Useful for users without local GPU.
- **Model selection per task** — Use a fast/small model for link filtering (simple classification) and a larger model for scoring (nuanced reasoning). Saves time without losing quality.
- **Structured output / tool use** — Use models that support structured JSON output natively (e.g., Ollama's `format: "json"` parameter) instead of parsing free-text responses.

### 6. Retry & Resilience

- **LLM call retries** — Wrap LLM calls in retry logic with exponential backoff (currently a single timeout crashes that job).
- **Circuit breaker per portal** — If a portal fails 3 times in a row, skip it for this run and flag it for review.
- **Checkpoint/resume** — Save pipeline state after each stage. If a run crashes at job #18/50 during scoring, resume from job #18 instead of starting over.

### 7. Scheduling

- **Cron / systemd timer** — Run the pipeline on a schedule (e.g., daily at 8 AM) and only surface new results.
- **Diff-based notifications** — Compare current run to previous run and highlight new jobs, removed jobs, and score changes.

---

## Feature Ideas

### For Individual Users (Power Tool)

#### Job Tracking & Workflow
- **Application tracker** — Mark jobs as "interested", "applied", "interviewing", "rejected", "offer". Track your pipeline.
- **Job notes** — Add personal notes to any job (e.g., "referred by X", "salary seems low").
- **Bookmark / shortlist** — Save top matches to a favorites list that persists across runs.
- **Hide / dismiss** — Permanently hide jobs you're not interested in so they don't appear in future runs.

#### Smarter Matching
- **Skill gap analysis** — For each job, show which required skills the user has vs. is missing. Helps prioritize upskilling.
- **Salary estimation** — Cross-reference job titles and locations with public salary data (levels.fyi, Glassdoor) to estimate compensation range.
- **Company enrichment** — Pull company info (size, funding, Glassdoor rating, tech stack) from public APIs and factor into scoring.
- **Location-aware matching** — Support "willing to relocate to X" or "remote-only" as hard filters, not just preferences.
- **Custom scoring weights** — Let users weight different factors (skills match: 40%, location: 20%, company size: 15%, domain: 25%).

#### Better Output
- **HTML report** — Generate a styled HTML page with job cards, scores, and reasoning. Easier to browse than a JSON file or terminal table.
- **Email digest** — Send a daily/weekly email summary of new matching jobs.
- **RSS feed** — Generate an RSS feed of new job matches that users can subscribe to in any reader.
- **Export to CSV/spreadsheet** — For users who want to manage their search in a spreadsheet.

#### Profile Intelligence
- **Resume parsing** — Instead of manually filling profile.yaml, upload a resume (PDF/DOCX) and auto-extract skills, experience, and preferences.
- **LinkedIn import** — Pull profile data from LinkedIn (via API or export).
- **Multi-profile support** — Maintain different profiles for different job searches (e.g., "backend roles" vs "ML roles") and compare results.

### For Multi-User / Product (Taking It to People)

#### Web Application
- **Dashboard UI** — React/Next.js frontend with:
  - Job browsing with filters (score, location, company, experience level)
  - Profile editor
  - Portal manager (add/remove career pages)
  - Run history and analytics
  - Application tracking board (Kanban-style)
- **User accounts** — Auth, personal profiles, saved searches
- **Mobile-responsive** — Browse job matches on phone

#### Backend Scaling
- **Task queue** — Use Celery + Redis (or similar) to process scraping and scoring jobs asynchronously. Web UI submits jobs, workers process them.
- **Shared scraping pool** — If multiple users track the same portal, scrape it once and share results. Massive efficiency gain.
- **API layer** — REST/GraphQL API so the pipeline can be triggered and consumed by any client (web, mobile, Slack bot, etc.).
- **Rate limiting** — Respect portal rate limits and robots.txt. Implement polite scraping with delays and user-agent rotation.

#### Community & Social
- **Portal directory** — Community-maintained list of career page URLs and their CSS selectors. Users contribute and vote on portal configs.
- **Job sharing** — Share a job listing with friends via link or message.
- **Company reviews integration** — Show Glassdoor/Blind reviews alongside job listings.

#### Monetization Ideas
- **Freemium model** — Free: 3 portals, 1 daily run. Paid: unlimited portals, hourly runs, email alerts, priority scraping.
- **Team/recruiter plan** — Recruiters can match candidate profiles against job listings in bulk.
- **API access** — Paid API for developers building on top of the job matching engine.

### Integrations

- **Slack/Discord bot** — `/jobify` command to trigger a run and get results in-channel.
- **Browser extension** — While browsing any career page, click to add it as a portal and get instant scoring.
- **Calendar integration** — Auto-add interview dates to Google Calendar / Outlook.
- **ATS integration** — One-click apply through Lever, Greenhouse, Workday APIs where available.
- **GitHub Actions** — Run the pipeline as a scheduled GitHub Action, commit results to a repo.

### Data & Analytics

- **Job market trends** — Track which skills are most in-demand over time, which companies are hiring most, salary trends by role/location.
- **Personal analytics** — "You've been scored highest for DevOps roles — consider focusing your search there."
- **Portal health monitoring** — Track which portals are returning good results vs. broken/empty ones. Auto-disable flaky portals.
- **A/B test scoring prompts** — Try different scoring prompts and compare which produces more actionable rankings.

---

## Additional Ideas

### 8. LLM Connection Pool

Instead of creating a new HTTP client per LLM call, maintain a pool of persistent connections to the Ollama server. This reduces connection overhead and enables true concurrent inference — multiple jobs can be scored simultaneously over pre-established connections rather than serially opening and closing one at a time.

### 9. Browser Worker Pool with Scrape Queue

Instead of a single browser visiting job pages one by one, run N Chromium instances (e.g., 2-3) as persistent workers that pull scrape jobs from a shared async queue:
- A `ScrapeQueue` holds all pending URLs (portal pages + individual job pages).
- Each browser worker runs in a loop: dequeue a URL, navigate, extract content, push result to a results queue.
- The pipeline submits URLs and awaits results without caring which browser handled it.
- Workers can be pre-warmed at pipeline start and reused across portals — no launch/close overhead per portal.
- If a worker crashes or hangs, it can be restarted without affecting others.
- Concurrency is tunable: 1 worker for a laptop, 5+ for a server with more RAM.
- This naturally decouples "what to scrape" from "how to scrape" — later you could swap Chromium workers for a headless API service (Browserless, Playwright cloud) without changing the pipeline.

### 10. LinkedIn Portal Discovery Service

A separate background service that logs into LinkedIn using the user's credentials and discovers career portals automatically. The goal is not to find jobs on LinkedIn itself, but to find companies that are hiring (from job posts, "we're hiring" updates, recruiter activity) and extract their actual career page URLs. These discovered portals get auto-added to the portal list for the main scraping pipeline.

### 11. AI Gateway Service

A standalone microservice that sits between Jobify and multiple LLM providers. It manages API keys for different providers (Ollama instances, Claude, OpenAI, Gemini, etc.) and distributes requests across them. Benefits:
- **Load balancing** — No single LLM server gets overloaded. Round-robin or least-connections routing.
- **Provider failover** — If one provider is down or slow, requests automatically go to another.
- **Cost optimization** — Route cheap tasks (link filtering) to free/local models and expensive tasks (scoring) to stronger cloud models.
- **Key rotation** — Manage multiple API keys per provider to stay within rate limits.

### 12. Persistent Job Catalog (Scrape Once, Query Forever)

Instead of re-visiting every career portal on each run, build a persistent job catalog:
- First visit: scrape the job page, extract structured details (title, skills required, experience range, description), and store in a database.
- Subsequent runs: only scrape the portal's listing page to detect new/removed job URLs. For known URLs, use the stored data directly — no need to re-visit and re-parse individual job pages.
- Periodically re-scrape stored jobs (e.g., weekly) to detect updates or removals.
- This dramatically reduces run time and LLM calls — a portal with 25 jobs that adds 2 new ones per week only needs 2 job page visits instead of 25.

---

## Technical Debt to Address First

1. **Error handling in matcher.py** — Wrap `parse_job_page` and `score_job` in try/except so a single LLM timeout doesn't crash the pipeline.
2. **Configurable LLM settings** — Move Ollama host, port, and model to environment variables or config file.
3. **Logging** — Replace `print`-based status callbacks with proper `logging` module. Add log levels, file output, and structured logs.
4. **Tests** — Unit tests for JSON parsing, experience filtering logic, URL normalization. Integration tests with mocked LLM responses.
5. **Type safety** — Add `py.typed` marker, run `mypy` in CI. The `callable` annotation on `on_status` should be `Callable[[str], None] | None`.
6. **Respect robots.txt** — Check `robots.txt` before scraping. Some career pages may disallow automated access.
