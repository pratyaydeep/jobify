import asyncio
from playwright.async_api import async_playwright
import json
import os
import re

def read_profile(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

async def get_rendered_text_playwright(url: str) -> str:
    """
    Uses Playwright to get the final, user-visible text from a page,
    then aggressively cleans it to isolate the job description.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, timeout=60000, wait_until='domcontentloaded')
            await asyncio.sleep(5) # Give page some time to render completely

            print("  - Extracting rendered text using Playwright inner_text('body')...")
            job_description_text = await page.inner_text('body')

            # Aggressive cleaning of common boilerplate
            boilerplate_patterns = [
                r'Skip to content',
                r'The following navigation element is controlled via arrow keys followed by tab',
                r'About TeamAmex', r'Career Areas', r'Locations', r'Colleague Networks',
                r'Students', r'Jobs', r'English', r'English-UK', r'Deutsch', r'Español', r'Français', r'中文', r'日本語',
                r'undefinedundefined', r'Loading\.\.\.', r'Careers',
                r'MANAGE YOUR PROFILE', r'Candidates', r'Colleagues', r'ABOUT',
                r'About American Express', r'Investor Relations', r'Follow American Express on',
                r'Visit American Express Site Map AdChoices Privacy Statement Recruitment Fraud Warning Terms of Service AI Usage Contact',
                r'This site is protected by reCAPTCHA and the Google Privacy Policy and Terms of Service apply\.',
                r'View all jobs', r'Similar jobs',
                r'Powered by .*', # e.g., Powered by Eightfold.ai
                r'American Express is an equal opportunity employer and makes employment decisions without regard to race, color, religion, sex, sexual orientation, gender identity, national origin, veteran status, disability status, or any other status protected by law\.',
                r'Find jobs at .*',
                r'Search by keyword', r'Search by location',
                r'Back to .*',
                r'Apply Now', r'Save Job'
            ]
            
            print("  - Applying aggressive boilerplate removal...")
            for pattern in boilerplate_patterns:
                job_description_text = re.sub(pattern, '', job_description_text, flags=re.IGNORECASE)

            return job_description_text.strip()
        except Exception as e:
            print(f"Error in get_rendered_text_playwright for {url}: {e}")
            return ""
        finally:
            await browser.close()

async def generate_match_prompt_payload(profile_content, job_url):
    print(f"Fetching and rendering job page from {job_url} using Playwright...")
    job_description = await get_rendered_text_playwright(job_url)

    if not job_description:
        print(f"Failed to fetch rendered text from {job_url}")
        return

    # Basic cleaning: reduce excessive newlines and spaces
    job_description = re.sub(r'\n\s*\n+', '\n', job_description) # Reduce multiple newlines to single
    job_description = re.sub(r'(\s){2,}', r'\1', job_description)  # Reduce multiple spaces/newlines to single
    job_description = job_description.strip()


    prompt = f"""
    Is the following job a good match for the profile provided?
    Respond with only "yes" or "no".

    PROFILE:
    ---
    {profile_content}
    ---

    JOB DESCRIPTION:
    ---
    {job_description}
    ---
    """
    
    payload = {
        "prompt": prompt,
        "stream": False,
        "n_predict": 10,
        "temperature": 0.1
    }

    with open('match_test.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    
    print("Generated match_test.json with the new, shorter payload.")
    print("\n--- CLEANED JOB DESCRIPTION (first 500 chars) ---")
    print(job_description[:500])
    print("\n-------------------------------------------------")


if __name__ == '__main__':
    profile_path = 'profile.txt'
    test_job_url = "https://aexp.eightfold.ai/careers/job/38953745"
    
    profile_data = read_profile(profile_path)
    asyncio.run(generate_match_prompt_payload(profile_data, test_job_url))