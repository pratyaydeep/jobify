# Jobify

Automated job search tool that scrapes company career pages, extracts job listings, and uses a local LLM (via Ollama) to match and rank jobs against your profile.

## How it works

1. **Scrape** — Uses Playwright to load each career page, scrolls to reveal all content, clicks "Load more" buttons, and extracts all links
2. **Filter** — Sends the scraped links to the LLM to identify which ones are actual job postings
3. **Parse** — Visits each job posting page and uses the LLM to extract title, location, and description
4. **Score** — Compares each job against your profile and assigns a relevance score (1-10) with reasoning
5. **Output** — Displays a ranked table and saves results to `output/suggestions.json`

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- [Ollama](https://ollama.com/) running with the `gemma3n` model

Pull the model if you haven't already:

```bash
ollama pull gemma3n
```

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd jobify

# Install dependencies
uv sync

# Install Playwright's Chromium browser
uv run playwright install chromium
```

## Configuration

### Profile

Edit `config/profile.yaml` with your details:

```yaml
name: "Your Name"
title: "Your Job Title"
experience_years: 5

skills:
  - Python
  - JavaScript
  - AWS

preferences:
  location: "Remote"
  role_type: "Full-time"
  domain: "Developer tools, AI/ML"

summary: |
  A brief summary of your background and what you're looking for.
```

### Career Portals

Edit `config/portals.yaml` with the company career pages you want to monitor:

```yaml
portals:
  - name: "Anthropic"
    url: "https://www.anthropic.com/careers"

  - name: "Stripe"
    url: "https://stripe.com/jobs/search"

  # Optional: use a CSS selector to narrow the scrape area
  - name: "Example Corp"
    url: "https://example.com/careers"
    selector: "#job-listings"
```

## Usage

```bash
# Run the full job search pipeline
uv run jobify scrape

# View configured career portals
uv run jobify portals

# View your current profile
uv run jobify profile
```

### Output

- `data/jobs.json` — All extracted job listings (raw)
- `output/suggestions.json` — Jobs ranked by match score with reasoning

## Project Structure

```
jobify/
├── config/
│   ├── profile.yaml        # Your profile
│   └── portals.yaml        # Career portals to scrape
├── src/jobify/
│   ├── cli.py              # CLI entry point
│   ├── scraper.py          # Playwright-based scraper with pagination
│   ├── llm.py              # Ollama API client (extraction + scoring)
│   ├── matcher.py          # Pipeline orchestration
│   └── models.py           # Data models
├── data/                   # Scraped job data
└── output/                 # Ranked suggestions
```

## Notes

- The Ollama server is configured to connect to `wayne:11434`. To change this, edit `OLLAMA_BASE` in `src/jobify/llm.py`.
- Each portal is capped at 25 job links to keep run times reasonable.
- The pipeline handles timeouts and errors gracefully — if one job page fails, it skips it and continues.
