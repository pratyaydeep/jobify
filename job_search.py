import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin, urlparse, parse_qs
import asyncio
import aiofiles
import httpx
from datetime import datetime
import re
from playwright.async_api import async_playwright, Browser, TimeoutError as PlaywrightTimeoutError
import os

# --- Configuration ---
LLM_SERVER_URL = "http://wayne.local:6001/completion"
MAX_DEPTH = 0 # Keep depth low for testing with Playwright
ATS_DOMAINS = [
    'taleo.net', 'greenhouse.io', 'lever.co', 'workday.com', 
    'icims.com', 'eightfold.ai', 'myworkdayjobs.com'
]

async def log_llm_response(prompt, response):
    """Asynchronously logs LLM prompts and responses to llm_responses.log."""
    timestamp = datetime.now().isoformat()
    log_entry = {"timestamp": timestamp, "prompt": prompt, "response": response}
    async with aiofiles.open('llm_responses.log', 'a') as f:
        await f.write(json.dumps(log_entry) + "\n")

async def log_detailed_llm_interaction(prompt, payload, full_response_content, error=None):
    """Asynchronously logs detailed LLM request/response to all_llm_requests.log."""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "prompt_sent": prompt,
        "request_payload": payload,
        "raw_response_received": full_response_content,
        "error": str(error) if error else None
    }
    async with aiofiles.open('all_llm_requests.log', 'a') as f:
        await f.write(json.dumps(log_entry, indent=2) + "\n---\n")

async def _call_llm_server(prompt: str, stream: bool = False, timeout: float = 600.0) -> str:
    """Generic helper to call the Llama.cpp server with a given prompt."""
    payload = {
        "prompt": prompt,
        "stream": stream,
        "n_predict": -1,
        "temperature": 0.1
    }
    
    error_occurred = None
    json_response = {}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(LLM_SERVER_URL, json=payload)
            response.raise_for_status()
            
            json_response = response.json()
            content = json_response.get("content", "")
            
        return content
    except httpx.ReadTimeout:
        error_message = f"LLM call timed out after {timeout} seconds."
        print(error_message)
        error_occurred = error_message
        return ""
    except httpx.HTTPStatusError as e:
        error_message = f"HTTP error calling LLM server: {e}"
        print(error_message)
        error_occurred = error_message
        return ""
    except Exception as e:
        error_message = f"Error calling LLM server: {e}"
        print(error_message)
        error_occurred = error_message
        return ""
    finally:
        # We log the raw response object in the detailed log
        await log_detailed_llm_interaction(prompt, payload, json_response, error_occurred)


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
        await page.goto(url, timeout=30000)
        await asyncio.sleep(5)
        content = await page.content()
        return content
    except PlaywrightTimeoutError:
        print(f"  - Timeout error while loading {url}.")
        if page:
            return await page.content()
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

    MAX_LLM_LINKS = 50
    if len(links) > MAX_LLM_LINKS:
        print(f"  - Warning: Truncating {len(links)} links to {MAX_LLM_LINKS} for LLM classification.")
        links = links[:MAX_LLM_LINKS]

    if not links:
        return [], []

    prompt = f"""You are an expert data extractor. Your task is to classify a list of URLs.
You must respond ONLY with a single, valid JSON object. Do not add any conversational text, explanations, or markdown code blocks.

The JSON object must have the following structure:
{{
  "job_post": ["<A_URL_that_is_a_direct_link_to_a_job_description>"],
  "navigation": ["<A_URL_that_leads_to_another_page_of_job_listings>"]
}}

---

Classify the following URLs from the source URL: {page_url}

{json.dumps(links, indent=2)}
"""
    
    print("--- Calling LLM for link classification ---")
    llm_content = await _call_llm_server(prompt)

    if not llm_content:
        return [], []

    try:
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", llm_content)
        if json_match:
            json_string = json_match.group(1).strip()
        else:
            # If no markdown is found, assume the whole content is the JSON string
            json_string = llm_content.strip()

        classified_links = json.loads(json_string)
        
        await log_llm_response(prompt, json.dumps(classified_links))
        return classified_links.get('job_post', []), classified_links.get('navigation', [])
    except Exception as e:
        error_message = f"Error processing LLM JSON response for {page_url}: {e}"
        print(error_message)
        await log_llm_response(prompt, f"{error_message}, Raw response: {llm_content}")
        return [], []

async def match_job_description(job_description, profile):
    # Properly escape job_description to avoid JSON errors in the payload
    escaped_job_description = json.dumps(job_description)
    
    prompt = f"""
    Is the following job a good match for the profile provided?
    Respond with only "yes" or "no".

    PROFILE:
    ---
    {profile}
    ---

    JOB DESCRIPTION:
    ---
    {escaped_job_description}
    ---
    """
    print("--- LLM Prompt for match_job_description ---")
    print(prompt)
    print("---------------------------------------------")
    
    print("--- Calling LLM for job matching ---")
    response = await _call_llm_server(prompt, timeout=180.0) # Shorter timeout for yes/no
    is_match = response.strip().lower().startswith('yes')
    await log_llm_response(prompt, response)
    return is_match

async def main():
    if os.path.exists("prompt_test.json"):
        os.remove("prompt_test.json")
        
    profile = read_profile('profile.txt')
    initial_pages = read_career_pages('career_pages.txt')
    
    queue = [(url, 0) for url in initial_pages]
    visited_urls = set()
    all_found_jobs = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        # --- Discovery Phase ---
        print("--- " + " Starting Link Discovery Phase ---")
        while queue:
            url, depth = queue.pop(0)
            
            if url in visited_urls or depth > MAX_DEPTH:
                continue

            print(f"[Depth {depth}] Discovering links on: {url}")
            visited_urls.add(url)
            
            html_content = await get_page_content(url, browser)
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
                
                job_content_html = await get_page_content(job_url, browser)
                if job_content_html:
                    job_soup = BeautifulSoup(job_content_html, 'html.parser')
                    job_text = job_soup.get_text(separator=' ', strip=True)
                    if not job_text.strip():
                        print("    -> Job description is empty, skipping.")
                        continue
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
