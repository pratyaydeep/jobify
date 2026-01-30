import requests
from bs4 import BeautifulSoup
import ollama
import json
from urllib.parse import urljoin, urlparse, parse_qs
import asyncio
import aiofiles
import httpx
from datetime import datetime
import re
from playwright.async_api import async_playwright, Browser, TimeoutError as PlaywrightTimeoutError

# --- Configuration ---
MAX_DEPTH = 1 # Keep depth low for testing with Playwright
ATS_DOMAINS = [
    'taleo.net', 'greenhouse.io', 'lever.co', 'workday.com', 
    'icims.com', 'eightfold.ai', 'myworkdayjobs.com'
]

async def log_llm_response(prompt, response):
    """Asynchronously logs LLM prompts and responses to a file."""
    timestamp = datetime.now().isoformat()
    log_entry = {"timestamp": timestamp, "prompt": prompt, "response": response}
    async with aiofiles.open('llm_responses.log', 'a') as f:
        await f.write(json.dumps(log_entry) + "\n")

def read_profile(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def read_career_pages(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

async def get_page_content(url: str, browser: Browser) -> str | None:
    """Asynchronously fetches and returns the full, JS-rendered HTML content of a URL using Playwright."""
    print(f"  - Navigating with browser to: {url}")
    page = None # Initialize page outside try block
    try:
        page = await browser.new_page()
        await page.goto(url, timeout=30000) # Removed wait_until
        await asyncio.sleep(5) # Increased sleep
        content = await page.content()
        return content
    except PlaywrightTimeoutError:
        print(f"  - Timeout error while loading {url}.")
        if page:
            return await page.content() # Return what we have so far
        return None
    except Exception as e:
        print(f"  - Error scraping {url} with Playwright: {e}")
        return None
    finally:
        if page:
            await page.close()


def get_main_domain(page_url):
    """Extracts the main domain from a URL, checking for a 'domain' query parameter first."""
    try:
        parsed_url = urlparse(page_url)
        query_params = parse_qs(parsed_url.query)
        if 'domain' in query_params:
            return query_params['domain'][0]
        
        parts = parsed_url.netloc.split('.')
        if len(parts) > 2:
            return '.'.join(parts[-2:])
        return parsed_url.netloc
    except Exception:
        return ""


async def classify_links_with_llm(html_content, page_url):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    a_tag_links = [urljoin(page_url, a['href']) for a in soup.find_all('a', href=True)]
    
    json_links = []
    smart_apply_data = soup.find('code', id='smartApplyData')
    if smart_apply_data:
        try:
            data = json.loads(smart_apply_data.string)
            if 'positions' in data and isinstance(data['positions'], list):
                for position in data['positions']:
                    if 'canonicalPositionUrl' in position:
                        json_links.append(position['canonicalPositionUrl'])
        except (json.JSONDecodeError, TypeError):
            print(f"  - Could not parse embedded JSON data on {page_url}")

    all_links = set(a_tag_links + json_links)
    
    main_domain = get_main_domain(page_url)
    page_domain = urlparse(page_url).netloc

    filtered_links = []
    for link in all_links:
        try:
            parsed_link = urlparse(link)
            if parsed_link.scheme not in ['http', 'https']:
                continue

            link_domain = parsed_link.netloc
            if link_domain == page_domain: # Strict same-domain filtering
                filtered_links.append(link)
        except Exception:
            continue
            
    links = sorted(list(set(filtered_links)))

    # Limit the number of links sent to the LLM to prevent long prompts and timeouts
    MAX_LLM_LINKS = 50
    if len(links) > MAX_LLM_LINKS:
        print(f"  - Warning: Truncating {len(links)} links to {MAX_LLM_LINKS} for LLM classification.")
        links = links[:MAX_LLM_LINKS]

    if not links:
        return [], []

    prompt = f"""
    You are a web scraping data processor. Your task is to classify URLs. Do not explain your reasoning.
    The source URL is: {page_url}
    You are provided with a list of links that are all from the exact same domain as the source URL.
    
    From this list, classify URLs into two categories:
    1. 'job_post': A direct link to a specific job description.
    2. 'navigation': A link to another page of job listings (e.g., "next", "page 2", "more jobs").

    Return ONLY a single, valid JSON object wrapped in a markdown code block.
    Example: ```json\n{{"job_post": ["URL1"], "navigation": ["URL2"]}}\n```
    
    Here is the list of links to classify:
    {json.dumps(links, indent=2)}
    """
    
    print("--- LLM Prompt for classify_links_with_llm ---")
    print(prompt)
    print("---------------------------------------------")

    full_response_content = "" # Initialize before try block

    async def _generate_and_accumulate_classify():
        nonlocal full_response_content
        async for chunk in await ollama.AsyncClient(host="http://wayne.local:11434").generate(
            model="gemma3:27b",
            prompt=prompt,
            stream=True
        ):
            full_response_content += chunk.get('response', '')

    try:
        await asyncio.wait_for(_generate_and_accumulate_classify(), timeout=600.0)
        
        await log_llm_response(prompt, full_response_content)
        
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", full_response_content)
        if not json_match:
            print(f"  - No JSON markdown block found in LLM response for {page_url}.")
            return [], []
            
        json_string = json_match.group(1)
        classified_links = json.loads(json_string)
        
        return classified_links.get('job_post', []), classified_links.get('navigation', [])
    except asyncio.TimeoutError:
        error_message = f"Error processing LLM response for {page_url}: Ollama call timed out after 180 seconds."
        print(error_message)
        await log_llm_response(prompt, f"{error_message}, Raw response: {full_response_content}")
        return [], []
    except Exception as e:
        error_message = f"Error processing LLM response for {page_url}: {e}"
        print(error_message)
        await log_llm_response(prompt, f"{error_message}, Raw response: {full_response_content}")
        return [], []

async def match_job_description(job_description, profile):
    prompt = f"""
    Is the following job a good match for the profile provided?
    Respond with only "yes" or "no".

    PROFILE:
    ---
    {profile}
    ---

    JOB DESCRIPTION:
    ---
    {job_description}
    ---
    """
    
    full_response_content = "" # Initialize before try block

    async def _generate_and_accumulate_match():
        nonlocal full_response_content
        async for chunk in ollama.AsyncClient(host="http://wayne.local:11434").generate(
            model="gemma3:27b",
            prompt=prompt,
            stream=True
        ):
            full_response_content += chunk.get('response', '')

    try:
        await asyncio.wait_for(_generate_and_accumulate_match(), timeout=180.0)
        
        await log_llm_response(prompt, full_response_content)
        return full_response_content.strip().lower() == 'yes'
    except asyncio.TimeoutError:
        error_message = f"Error during qualification matching: Ollama call timed out after 180 seconds."
        print(error_message)
        await log_llm_response(prompt, f"{error_message}, Raw response: {full_response_content}")
        return False
    except Exception as e:
        error_message = f"Error during qualification matching: {e}"
        print(error_message)
        await log_llm_response(prompt, f"{error_message}, Raw response: {full_response_content}")
        return False

async def main():
    profile = read_profile('profile.txt')
    initial_pages = read_career_pages('career_pages.txt')
    
    queue = [(url, 0) for url in initial_pages]
    visited_urls = set()
    all_found_jobs = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True) # Launch browser once

        # --- Discovery Phase ---
        print("--- " + " Starting Link Discovery Phase ---")
        while queue:
            url, depth = queue.pop(0)
            
            if url in visited_urls or depth > MAX_DEPTH:
                continue

            print(f"[Depth {depth}] Discovering links on: {url}")
            visited_urls.add(url)
            
            html_content = await get_page_content(url, browser) # Pass browser to get_page_content
            if not html_content:
                continue

            job_posts, navigation_links = await classify_links_with_llm(html_content, url)
            
            for job_url in job_posts:
                if job_url not in all_found_jobs:
                    print(f"  -> Discovered job link: {job_url}")
                    all_found_jobs.add(job_url)

            for nav_url in navigation_links:
                if nav_url not in visited_urls and nav_url not in all_found_jobs:
                    print(f"  -> Queueing navigation link: {nav_url}")
                    queue.append((nav_url, depth + 1))
        
        # --- Matching Phase ---
        print("\n---" + " Starting Matching Phase ---")
        matching_jobs = []
        async with aiofiles.open('all_jobs.txt', 'w') as all_jobs_file:
            for i, job_url in enumerate(all_found_jobs):
                print(f"  ({i+1}/{len(all_found_jobs)}) Analyzing job: {job_url}")
                await all_jobs_file.write(f"{job_url}\n")
                
                job_content_html = await get_page_content(job_url, browser) # Pass browser to get_page_content
                if job_content_html:
                    job_soup = BeautifulSoup(job_content_html, 'html.parser')
                    job_text = job_soup.get_text(separator=' ', strip=True)
                    if await match_job_description(job_text, profile):
                        print(f"    -> MATCH FOUND!")
                        matching_jobs.append(job_url)

        await browser.close()

    async with aiofiles.open('matching_jobs.txt', 'w') as f:
        await f.write("\n".join(matching_jobs))
    
    print("\nJob search complete.")
    print(f"Discovered {len(all_found_jobs)} total job links.")
    print(f"Found {len(matching_jobs)} matching jobs.")

if __name__ == '__main__':
    asyncio.run(main())
