from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from jobify.matcher import load_portals, load_profile, run_pipeline

console = Console()

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging to file and optionally to stderr."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.FileHandler(LOG_DIR / "jobify.log"),
    ]
    if verbose:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def cmd_scrape(args: argparse.Namespace) -> None:
    """Run the full scrape -> extract -> match pipeline."""
    profile = load_profile()
    portals = load_portals()

    console.print(f"\n[bold]Profile:[/bold] {profile.name} — {profile.title}")
    console.print(f"[bold]Portals:[/bold] {len(portals)} configured\n")

    with console.status("[bold green]Running job search pipeline...") as status_spinner:

        def on_status(msg: str) -> None:
            status_spinner.update(f"[bold green]{msg}")
            console.log(msg)

        scored = asyncio.run(run_pipeline(profile, portals, on_status=on_status))

    if not scored:
        console.print("[yellow]No jobs found. Check your portal URLs and try again.")
        return

    # Display results table
    table = Table(title=f"\nTop Job Suggestions ({len(scored)} total)")
    table.add_column("#", style="dim", width=4)
    table.add_column("Score", justify="center", width=6)
    table.add_column("Title", min_width=25)
    table.add_column("Company", min_width=15)
    table.add_column("Location", min_width=15)
    table.add_column("Reasoning", min_width=30)

    for i, s in enumerate(scored[:20], 1):
        score_color = "green" if s.score >= 7 else "yellow" if s.score >= 4 else "red"
        table.add_row(
            str(i),
            f"[{score_color}]{s.score:.0f}/10[/{score_color}]",
            s.job.title,
            s.job.company,
            s.job.location,
            s.reasoning[:80],
        )

    console.print(table)
    console.print(
        f"\n[dim]Full results saved to output/suggestions.json[/dim]\n"
    )


def cmd_portals(args: argparse.Namespace) -> None:
    """List configured career portals."""
    portals = load_portals()
    table = Table(title="Configured Career Portals")
    table.add_column("Name", min_width=15)
    table.add_column("URL", min_width=40)
    table.add_column("Selector", min_width=15)

    for p in portals:
        table.add_row(p.name, p.url, p.selector or "—")

    console.print(table)


def cmd_profile(args: argparse.Namespace) -> None:
    """Show the current profile."""
    profile = load_profile()

    console.print(f"\n[bold]Name:[/bold]       {profile.name}")
    console.print(f"[bold]Title:[/bold]      {profile.title}")
    console.print(f"[bold]Experience:[/bold] {profile.experience_years} years")
    console.print(f"[bold]Skills:[/bold]     {', '.join(profile.skills)}")
    console.print(f"[bold]Preferences:[/bold]")
    for k, v in profile.preferences.items():
        console.print(f"  {k}: {v}")
    console.print(f"[bold]Summary:[/bold]\n  {profile.summary.strip()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="jobify",
        description="Automated job search — scrape career portals, match with your profile using a local LLM",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging to stderr",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("scrape", help="Run the full job search pipeline")
    sub.add_parser("portals", help="List configured career portals")
    sub.add_parser("profile", help="Show your current profile")

    args = parser.parse_args()
    _setup_logging(verbose=args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "scrape": cmd_scrape,
        "portals": cmd_portals,
        "profile": cmd_profile,
    }
    commands[args.command](args)
