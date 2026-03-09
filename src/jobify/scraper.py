from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from urllib.parse import urljoin

from playwright.async_api import Page, async_playwright

from jobify.models import Portal

logger = logging.getLogger(__name__)


@dataclass
class ScrapedLink:
    text: str
    href: str


@dataclass
class ScrapedJob:
    """Raw scraped data for a single job posting."""

    link: ScrapedLink
    page_text: str


async def _scroll_to_bottom(page: Page, max_scrolls: int = 15) -> None:
    """Scroll the page to the bottom to trigger lazy-loaded content.

    Keeps scrolling until the page height stops changing or max_scrolls is hit.
    """
    prev_height = 0
    for _ in range(max_scrolls):
        curr_height = await page.evaluate("document.body.scrollHeight")
        if curr_height == prev_height:
            break
        prev_height = curr_height
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(1.5)


async def _click_load_more(page: Page, max_clicks: int = 5) -> None:
    """Try to click common 'Load more' / 'Show more' buttons."""
    patterns = [
        "text=/load more/i",
        "text=/show more/i",
        "text=/view more/i",
        "text=/see all/i",
        "text=/more jobs/i",
    ]
    for _ in range(max_clicks):
        clicked = False
        for selector in patterns:
            btn = page.locator(selector).first
            if await btn.is_visible(timeout=1000):
                await btn.click()
                await asyncio.sleep(2)
                clicked = True
                break
        if not clicked:
            break


async def _extract_links(page: Page, base_url: str) -> list[ScrapedLink]:
    """Extract all <a> links from the page with their text and resolved href."""
    raw_links = await page.evaluate("""
        () => Array.from(document.querySelectorAll('a[href]')).map(a => ({
            text: a.innerText.trim(),
            href: a.href
        }))
    """)
    links: list[ScrapedLink] = []
    seen: set[str] = set()
    for item in raw_links:
        href = item.get("href", "")
        text = item.get("text", "")
        if not href or not text or len(text) < 3:
            continue
        # Resolve relative URLs
        full_url = urljoin(base_url, href)
        if full_url in seen:
            continue
        seen.add(full_url)
        links.append(ScrapedLink(text=text, href=full_url))
    return links


async def _dismiss_cookie_banner(page: Page) -> None:
    """Try to click common cookie accept buttons."""
    for selector in ["text=/Accept/i", "text=/Accept all/i", "text=/Got it/i"]:
        try:
            btn = page.locator(selector).first
            if await btn.is_visible(timeout=1500):
                await btn.click()
                await asyncio.sleep(1)
                return
        except Exception:
            continue


async def scrape_job_page(page: Page, url: str, timeout_ms: int = 30000) -> str:
    """Visit a single job posting page and return its text content."""
    try:
        logger.debug("Visiting job page: %s", url)
        await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        await _dismiss_cookie_banner(page)
        await asyncio.sleep(3)
        return await page.inner_text("body")
    except Exception as exc:
        logger.warning("Failed to scrape job page %s: %s", url, exc)
        return ""


async def scrape_portal(
    portal: Portal, timeout_ms: int = 30000
) -> tuple[list[ScrapedLink], str]:
    """Scrape a career portal: scroll, paginate, and extract all links.

    Returns (all_links, page_text_for_context).
    """
    logger.info("Scraping portal: %s (%s)", portal.name, portal.url)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto(portal.url, wait_until="networkidle", timeout=timeout_ms)

        # Expand the page: scroll and click load-more buttons
        await _scroll_to_bottom(page)
        await _click_load_more(page)
        await _scroll_to_bottom(page)

        # Extract all links
        links = await _extract_links(page, portal.url)
        logger.info("Extracted %d links from %s", len(links), portal.name)

        # Also grab page text for context (helps LLM filter links)
        if portal.selector:
            el = await page.query_selector(portal.selector)
            page_text = await el.inner_text() if el else await page.inner_text("body")
        else:
            page_text = await page.inner_text("body")

        await browser.close()

    return links, page_text


async def scrape_job_pages(
    job_urls: list[str], timeout_ms: int = 20000
) -> dict[str, str]:
    """Visit each job URL and return {url: page_text}."""
    logger.info("Scraping %d job pages", len(job_urls))
    results: dict[str, str] = {}
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()

        for url in job_urls:
            text = await scrape_job_page(page, url, timeout_ms)
            results[url] = text

        await browser.close()
    return results
