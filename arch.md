# Jobify Architecture

---

## Microservice Design

### Service Decomposition

The current monolith has clear boundaries that map naturally to services:

```
Current monolith:  CLI → matcher → scraper → llm → JSON files
                                      ↓
Microservices:     Gateway → Orchestrator → Scraper Service
                                          → AI Service
                                          → Job Store
                                          → Profile Service
                                          → Notification Service
                                          → Portal Discovery Service
                                          → Analytics Service
```

---

### Services

#### 1. API Gateway
**Responsibility:** Single entry point for all clients (web, mobile, CLI, Slack bot).

- Routes requests to the correct service
- Handles authentication (JWT validation)
- Rate limiting per user
- Request/response logging
- WebSocket endpoint for real-time pipeline progress

**Tech:** FastAPI or Kong/Traefik

**Endpoints:**
- `POST /pipeline/run` — Trigger a new pipeline run
- `GET /pipeline/status/{run_id}` — Stream progress via SSE/WebSocket
- `GET /jobs` — Query job catalog
- `GET /suggestions/{profile_id}` — Get scored results
- `CRUD /profiles`, `CRUD /portals`

---

#### 2. Orchestrator Service
**Responsibility:** Coordinates the pipeline stages. The brain that knows the workflow but doesn't do the work itself.

- Receives a "run pipeline" request
- For each portal: publishes scrape tasks → waits for results → publishes LLM filter tasks → publishes individual job scrape tasks → publishes parse tasks → publishes score tasks
- Tracks run state (which stage, how many jobs processed, errors)
- Saves final results

**Communicates via:** RabbitMQ queues (publishes tasks, consumes results)

**State:** Run metadata in PostgreSQL (`runs` table with status, timestamps, counts)

This is the current `matcher.py` — but instead of calling functions directly, it publishes messages and reacts to results.

---

#### 3. Scraper Service
**Responsibility:** Headless browser management. Takes a URL, returns page content.

- Maintains a pool of N Chromium instances
- Consumes from `scrape_jobs` RabbitMQ queue
- Handles pagination, scrolling, cookie dismissal, lazy loading
- Returns raw page text + extracted links to a results queue
- Health checks on browser instances (restart if hung)

**Consumes:** `queue:scrape_jobs` — `{url, type: "portal"|"job_page", portal_name}`
**Publishes:** `queue:scrape_results` — `{url, links[], page_text, status}`

**Scaling:** Horizontally — run more instances with more browsers. Each instance manages its own browser pool. Stateless (no shared state between instances).

**Tech:** Python + Playwright, runs in Docker with Chromium pre-installed

---

#### 4. AI Service (LLM Gateway)
**Responsibility:** All LLM interactions. Abstracts away which model/provider is used.

- Consumes tasks from multiple queues: `filter_links`, `parse_job`, `score_job`
- Routes to appropriate LLM backend based on task type:
  - Link filtering → fast/cheap model (gemma3n local)
  - Job parsing → medium model
  - Job scoring → best available model
- Manages connections to multiple providers (Ollama, Claude API, OpenAI)
- API key rotation and failover
- Response caching via Redis (keyed by prompt hash)
- Retries with exponential backoff

**Consumes:**
- `queue:filter_links` — `{links[], portal_name}` → returns `{job_urls[]}`
- `queue:parse_job` — `{page_text, portal_name, url}` → returns `{Job}`
- `queue:score_job` — `{Job, Profile}` → returns `{score, reasoning}`

**Publishes:** Results back to corresponding result queues

**Scaling:** Add more instances pointing to different LLM servers. Each instance can hold a persistent connection pool to its assigned backend.

---

#### 5. Job Store Service
**Responsibility:** Persistent storage and querying of all job data.

- CRUD for jobs, scored jobs, and run history
- Deduplication (fuzzy match on company + title + location)
- Experience range filtering
- Full-text search across job descriptions
- Tracks job lifecycle: first_seen, last_seen, is_active
- Detects new vs. previously seen jobs per run

**Database:** PostgreSQL

**Tables:**
```
jobs           (id, url, title, company, location, description, min_exp, max_exp, source_portal, first_seen, last_seen, is_active, content_hash)
scored_jobs    (id, job_id, profile_id, score, reasoning, run_id, scored_at)
runs           (id, profile_id, started_at, completed_at, status, portals_scraped, jobs_found, jobs_scored)
```

**API:** gRPC or REST
- `POST /jobs` — Upsert a job (dedup by URL + content hash)
- `GET /jobs?min_exp=&max_exp=&portal=&search=` — Query with filters
- `GET /jobs/new?since=` — Jobs discovered since a given run
- `POST /scored-jobs` — Store scoring results

---

#### 6. Profile & Auth Service
**Responsibility:** User management, authentication, and profile storage.

- User registration/login (email + password, or OAuth)
- JWT token issuance and validation
- Multiple profiles per user (e.g., "backend roles", "ML roles")
- Profile CRUD
- Portal list management per user

**Database:** PostgreSQL

**Tables:**
```
users          (id, email, password_hash, created_at)
profiles       (id, user_id, name, title, experience_years, skills[], preferences, summary, is_default)
user_portals   (id, user_id, portal_name, portal_url, selector, is_active)
```

---

#### 7. Notification Service
**Responsibility:** Alerting users about new matches and pipeline completion.

- Consumes from `queue:notifications`
- Sends email digests (daily/weekly summary of new matches)
- Slack/Discord webhook integration
- Push notifications (if mobile app exists)
- User notification preferences (channels, frequency, minimum score threshold)

**Consumes:** `queue:notifications` — `{user_id, event_type, payload}`

**Tech:** Python + SendGrid/SES for email, Slack SDK for Slack

---

#### 8. Portal Discovery Service
**Responsibility:** Automatically find new career portals to scrape.

- LinkedIn integration: logs in, scans feed for hiring posts, extracts company career page URLs
- Google search integration: queries "company_name careers page" for verification
- Community-submitted portals (moderation queue)
- Validates discovered URLs (checks if they load, have job-like content)
- Publishes discovered portals for admin review

**Runs on:** Scheduled (daily cron), not triggered by user requests

---

#### 9. Analytics Service
**Responsibility:** Market intelligence and user insights.

- Consumes job events from Kafka
- Tracks skill demand trends over time
- Company hiring velocity
- Score distribution per user (which role types score highest)
- Portal health metrics (success rate, job count trends)

**Consumes:** Kafka topic `jobify.events`

**Stores:** Time-series data in PostgreSQL or ClickHouse

---

### Inter-Service Communication

```
┌─────────┐     REST/WS      ┌──────────────┐
│ Clients │◄────────────────►│  API Gateway  │
└─────────┘                  └──────┬───────┘
                                    │ REST/gRPC
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌──────────────┐ ┌───────────┐ ┌──────────────┐
            │ Profile/Auth │ │ Job Store │ │ Orchestrator │
            └──────────────┘ └───────────┘ └──────┬───────┘
                                                   │
                                            RabbitMQ queues
                                    ┌──────────┼──────────┐
                                    ▼          ▼          ▼
                              ┌──────────┐ ┌────────┐ ┌────────────┐
                              │ Scraper  │ │   AI   │ │Notification│
                              │ Service  │ │Service │ │  Service   │
                              │(N workers│ │(N wrkrs│ └────────────┘
                              └──────────┘ └───┬────┘
                                               │
                                          Redis Cache
                                               │
                                         Kafka Events
                                               │
                                    ┌──────────┴──────────┐
                                    ▼                     ▼
                              ┌───────────┐      ┌──────────────┐
                              │ Analytics │      │Portal Discov.│
                              └───────────┘      └──────────────┘
```

### Communication Patterns

| From → To | Method | Why |
|---|---|---|
| Client → Gateway | REST + WebSocket | Standard API access + real-time updates |
| Gateway → Profile/Auth | REST (sync) | Auth must complete before proceeding |
| Gateway → Job Store | REST (sync) | Direct query/response for job browsing |
| Orchestrator → Scraper | RabbitMQ (async) | Decoupled, retryable, parallelizable |
| Orchestrator → AI | RabbitMQ (async) | Same — fan-out to multiple workers |
| AI → Redis | Direct | Low-latency cache reads/writes |
| Orchestrator → Job Store | REST (sync) | Save results after each stage |
| Orchestrator → Notification | RabbitMQ (async) | Fire-and-forget, notification is non-blocking |
| All services → Kafka | Publish (async) | Event logging, analytics doesn't block pipeline |
| Kafka → Analytics | Consume (async) | Streaming aggregation at its own pace |

### Deployment

Each service runs as a separate Docker container, orchestrated with Docker Compose (dev) or Kubernetes (prod):

```yaml
services:
  gateway:        # 1 instance
  orchestrator:   # 1 instance
  scraper:        # 2-5 instances (each with browser pool)
  ai-service:     # 2-3 instances (each with LLM connections)
  job-store:      # 1 instance
  profile-auth:   # 1 instance
  notification:   # 1 instance
  analytics:      # 1 instance
  portal-discovery: # 1 instance (scheduled)

  # Infrastructure
  postgres:       # shared DB (separate logical DBs per service)
  redis:          # shared cache
  rabbitmq:       # message broker
  kafka:          # event streaming (optional, add later)
```

### Migration Path from Monolith

Don't rewrite everything at once. Extract services one at a time:

1. **Extract Scraper Service first** — It's the most resource-intensive and benefits most from independent scaling. The monolith publishes URLs to RabbitMQ, scraper workers consume them.

2. **Extract AI Service second** — Similar reasoning. LLM calls are slow and benefit from connection pooling and caching. Add Redis caching at this step.

3. **Add API Gateway + Job Store** — When building the web UI. The CLI can still call the orchestrator directly; the web UI goes through the gateway.

4. **Extract Profile/Auth** — When adding multi-user support.

5. **Add Notification, Analytics, Portal Discovery** — As features, not as migrations. These are new capabilities built as services from the start.

---

### Monorepo File Structure

Keep everything in one repo, run each service as a separate process:

```
jobify/
├── config/
│   ├── profile.yaml
│   ├── portals.yaml
│   └── settings.yaml              # NEW — shared config (Redis, RabbitMQ, Postgres, Ollama)
│
├── src/
│   ├── shared/                    # NEW — shared code across services
│   │   ├── __init__.py
│   │   ├── models.py              # MOVED from jobify/ — Job, Profile, ScoredJob, Portal
│   │   ├── schemas.py             # NEW — RabbitMQ message schemas (Pydantic)
│   │   ├── config.py              # NEW — load settings.yaml + env vars
│   │   ├── db.py                  # NEW — SQLAlchemy models + session factory
│   │   └── queue.py               # NEW — RabbitMQ publish/consume helpers
│   │
│   ├── gateway/                   # NEW — API Gateway service
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI app, routes, WebSocket for progress
│   │   ├── auth.py                # JWT middleware
│   │   └── run.py                 # uvicorn entrypoint
│   │
│   ├── orchestrator/              # REPLACES matcher.py
│   │   ├── __init__.py
│   │   ├── pipeline.py            # Pipeline state machine (publish tasks, react to results)
│   │   └── run.py                 # entrypoint — connects to RabbitMQ, starts pipeline
│   │
│   ├── scraper/                   # REPLACES scraper.py
│   │   ├── __init__.py
│   │   ├── browser_pool.py        # Chromium instance pool management
│   │   ├── handlers.py            # scrape_portal, scrape_job_page (current logic)
│   │   ├── pagination.py          # scroll, click load more, cookie dismiss
│   │   └── run.py                 # entrypoint — consumes from scrape_jobs queue
│   │
│   ├── ai_service/                # REPLACES llm.py
│   │   ├── __init__.py
│   │   ├── providers.py           # Ollama, Claude, OpenAI client wrappers
│   │   ├── prompts.py             # All prompt templates (filter, parse, score)
│   │   ├── parser.py              # JSON parsing helpers (_parse_json_array, etc.)
│   │   ├── cache.py               # Redis prompt-hash cache
│   │   └── run.py                 # entrypoint — consumes from filter/parse/score queues
│   │
│   ├── job_store/                 # NEW — replaces JSON file storage
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI app with CRUD endpoints
│   │   ├── repository.py          # DB queries (upsert, dedup, search, filter)
│   │   └── run.py                 # uvicorn entrypoint
│   │
│   ├── profile_service/           # NEW — user & profile management
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI app (CRUD profiles, portals, auth)
│   │   └── run.py                 # uvicorn entrypoint
│   │
│   ├── notification/              # NEW
│   │   ├── __init__.py
│   │   ├── channels.py            # email, slack, discord senders
│   │   └── run.py                 # entrypoint — consumes from notifications queue
│   │
│   └── cli/                       # REPLACES jobify/cli.py
│       ├── __init__.py
│       └── main.py                # CLI that talks to gateway API instead of calling pipeline directly
│
├── docker-compose.yaml            # NEW — run all services + infra
├── Dockerfile.scraper             # NEW — Chromium + Playwright
├── Dockerfile.service             # NEW — generic Python service
├── pyproject.toml                 # UPDATED — workspace with per-service deps
└── Makefile                       # NEW — convenience commands
```

### File Migration Map

| Current file | What happens | New location |
|---|---|---|
| `models.py` | Moves to shared | `src/shared/models.py` |
| `scraper.py` | Splits into 3 files | `src/scraper/handlers.py`, `browser_pool.py`, `pagination.py` |
| `llm.py` | Splits into 4 files | `src/ai_service/providers.py`, `prompts.py`, `parser.py`, `cache.py` |
| `matcher.py` | Becomes orchestrator | `src/orchestrator/pipeline.py` |
| `cli.py` | Becomes API client | `src/cli/main.py` |
| `data/jobs.json` | Replaced by PostgreSQL | `src/job_store/repository.py` |
| `output/suggestions.json` | Replaced by PostgreSQL | Queried via `GET /suggestions` |

### New Shared Files

**`src/shared/config.py`** — Single place for all service configuration:
```python
# Reads from settings.yaml + env var overrides
# REDIS_URL, RABBITMQ_URL, DATABASE_URL, OLLAMA_BASE, etc.
```

**`src/shared/schemas.py`** — Message contracts between services:
```python
# ScrapeTask(url, type, portal_name)
# ScrapeResult(url, links, page_text, status)
# FilterLinksTask(links, portal_name)
# ParseJobTask(page_text, portal_name, url)
# ScoreJobTask(job, profile)
```

**`src/shared/queue.py`** — RabbitMQ helpers:
```python
# publish(queue_name, message)
# consume(queue_name, handler_fn)
# Handles connection pooling, serialization, acks
```

**`src/shared/db.py`** — Database models:
```python
# SQLAlchemy ORM: JobRow, ScoredJobRow, RunRow, UserRow, ProfileRow
# Session factory, migrations via Alembic
```

### Running Services

**Development (separate terminals):**
```bash
# Start infrastructure
docker-compose up -d postgres redis rabbitmq

# Start services (each in its own terminal)
python -m src.gateway.run
python -m src.orchestrator.run
python -m src.scraper.run
python -m src.ai_service.run
python -m src.job_store.run
```

**With a Makefile:**
```bash
make infra        # docker-compose up -d postgres redis rabbitmq
make services     # starts all Python services in background
make scraper      # start just the scraper (for scaling extra instances)
make logs         # tail all service logs
```

**Production (Docker Compose):**
```yaml
# docker-compose.yaml
services:
  postgres:
    image: postgres:16
    volumes: [pgdata:/var/lib/postgresql/data]
  redis:
    image: redis:7
  rabbitmq:
    image: rabbitmq:3-management
  gateway:
    build: { dockerfile: Dockerfile.service }
    command: python -m src.gateway.run
    ports: ["8000:8000"]
  orchestrator:
    build: { dockerfile: Dockerfile.service }
    command: python -m src.orchestrator.run
  scraper:
    build: { dockerfile: Dockerfile.scraper }
    command: python -m src.scraper.run
    deploy: { replicas: 3 }
  ai-service:
    build: { dockerfile: Dockerfile.service }
    command: python -m src.ai_service.run
    deploy: { replicas: 2 }
  job-store:
    build: { dockerfile: Dockerfile.service }
    command: python -m src.job_store.run
  notification:
    build: { dockerfile: Dockerfile.service }
    command: python -m src.notification.run
```

The core logic (scraping, LLM prompts, JSON parsing) stays the same — it just gets wired through queues instead of direct function calls. The `shared/` package prevents code duplication across services.

---

# Infrastructure: Kafka, Redis & RabbitMQ in Jobify

Where message brokers and caches fit into the system at different stages of growth.

---

## Redis

### LLM Response Cache
Cache LLM responses keyed by a hash of the prompt. If the same job page is re-scraped and the text hasn't changed, skip the LLM call entirely. Use TTL-based expiry (e.g., 7 days) so stale entries clean themselves up.

### Job Dedup Set
Use `SADD seen_urls <url>` to instantly check if a job URL was already processed in a previous run. Much faster than querying a database for every URL.

### Rate Limit Counters
Use `INCR portal:visa:requests` with TTL to enforce polite scraping limits (e.g., max 10 requests/minute per portal). Prevents hammering career pages and getting blocked.

### Session Store
When multi-user auth is added, store session tokens in Redis instead of hitting the database on every request. Fast reads, automatic expiry.

### Real-Time Pipeline Progress
Use Redis pub/sub to push live pipeline status updates ("Scoring job 5/20...") to a web frontend via WebSockets. The CLI currently prints status — Redis makes that status available to any connected client.

---

## RabbitMQ

### Scrape Queue
The natural backend for the browser worker pool. The pipeline publishes URLs to a `scrape_jobs` queue, and N Chromium workers consume from it.

Benefits over a simple `asyncio.Queue`:
- **Acknowledgments** — If a worker crashes mid-scrape, the URL goes back to the queue automatically. No lost work.
- **Priority queues** — New portals can be prioritized over re-checks.
- **Prefetch limits** — Control how many URLs each worker takes at a time to prevent memory overload.
- **Persistence** — If the whole system restarts, unprocessed URLs survive in the queue.

### LLM Scoring Queue
Same pattern as scraping. Publish `(job, profile)` pairs to a `scoring_tasks` queue. Multiple LLM workers (each with their own Ollama connection or API key) consume and score in parallel. This is where the LLM connection pool and AI gateway ideas converge — RabbitMQ becomes the glue between them.

### Notification Queue
When scoring completes, publish to a `notifications` queue. Independent consumers can:
- Send email digests
- Post to Slack/Discord
- Push mobile notifications
- Update a web dashboard

Each consumer processes at its own pace without blocking the pipeline.

---

## Kafka

### Job Event Log
Every pipeline event becomes a Kafka message with a topic like `jobify.events`:
- `job.discovered` — New URL found on a portal
- `job.parsed` — LLM extracted structured data
- `job.scored` — Score assigned to a job
- `job.filtered_out` — Job excluded by experience filter
- `job.applied` — User marked as applied

This creates an immutable audit trail of everything the system has ever seen. Unlike RabbitMQ (where messages are consumed and gone), Kafka retains events for replay and analysis.

### Multi-Consumer Fan-Out
The same `job.discovered` event can be consumed by multiple independent consumer groups:
- **Scoring service** — Scores the job against the user's profile
- **Analytics service** — Tracks market trends (which skills are hot, which companies are hiring)
- **Dedup service** — Checks if this job was already seen from another portal
- **Notification service** — Alerts the user about new matches

Each consumer group processes independently and at its own pace. Adding a new consumer doesn't affect existing ones.

### Portal Change Detection
Stream portal scrape results to a `jobify.portal_snapshots` topic. A consumer compares the current snapshot to the previous one to detect:
- New job URLs added since last scrape
- Job URLs removed (position filled or delisted)
- Changes to existing job pages (updated description, new experience requirements)

The pipeline doesn't need to track this state itself — Kafka's retention and offset management handle it.

### Cross-User Analytics
When multiple users run Jobify, Kafka aggregates all job discovery events across users. This powers:
- **Skill demand trends** — "Python demand up 15% this month"
- **Company hiring velocity** — "Visa posted 12 new roles this week"
- **Salary benchmarking** — Aggregate inferred salary data across similar roles
- **Portal reliability scores** — Which portals consistently return good data vs. break often

---

## When to Introduce Each

| Scale | Redis | RabbitMQ | Kafka |
|---|---|---|---|
| **Single user, local** | LLM cache, dedup set | Not needed | Not needed |
| **Multi-user, small team** | + sessions, rate limits | Scrape queue, LLM queue | Not needed |
| **Product with many users** | + real-time progress | + notification queue | Event log, analytics, fan-out |

### Recommended Order of Adoption

1. **Redis first** — Immediate value with minimal setup. Caching LLM responses alone can cut repeat run times by 80%+. A single Redis instance handles all use cases at this stage.

2. **RabbitMQ second** — When you implement the browser worker pool and parallel LLM scoring. This is the point where simple `asyncio.Queue` stops being enough (no persistence, no acknowledgments, single-process only).

3. **Kafka last** — When you have multiple users generating enough event volume to make streaming analytics worthwhile. Kafka's operational overhead isn't justified until you need durable event logs and multi-consumer fan-out at scale.

---

## Architecture Diagram

```
                                    ┌─────────────┐
                                    │   Redis     │
                                    │ - LLM cache │
                                    │ - Dedup set │
                                    │ - Sessions  │
                                    │ - Pub/Sub   │
                                    └──────┬──────┘
                                           │
┌──────────┐    ┌─────────────┐    ┌───────┴───────┐    ┌──────────────┐
│  Portal  │───▶│  RabbitMQ   │───▶│   Workers     │───▶│   RabbitMQ   │
│  URLs    │    │ scrape_jobs │    │ Browser Pool  │    │ scoring_tasks│
└──────────┘    └─────────────┘    └───────────────┘    └──────┬───────┘
                                                               │
                                                       ┌───────┴───────┐
                                                       │   Workers     │
                                                       │  LLM Pool     │
                                                       └───────┬───────┘
                                                               │
                                                       ┌───────┴───────┐
                                                       │    Kafka      │
                                                       │ jobify.events │
                                                       └───────┬───────┘
                                                               │
                                          ┌────────────┬───────┴────┬──────────┐
                                          ▼            ▼            ▼          ▼
                                      Analytics   Notifications  Dedup    Dashboard
```
