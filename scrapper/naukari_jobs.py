# import time
# import random
# import pandas as pd
# import os
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
# from bs4 import BeautifulSoup
# from datetime import datetime

# # ── Config ──────────────────────────────────────────────
# JOB_TYPES = [
#     "machine-learning-jobs",
#     "artificial-intelligence-jobs",
#     "data-science-jobs",
#     "data-analyst-jobs",
#     "python-developer-jobs",
#     "data-engineer-jobs",
#     "deep-learning-jobs",
#     "nlp-jobs",
#     "computer-vision-jobs",
#     "generative-ai-jobs",
# ]
# JOBS_PER_TYPE = 30
# PAGES_PER_TYPE = 5
# OUTPUT_PATH = "data/raw/naukri_jobs.csv"
# BASE_URL = "https://www.naukri.com"
# # ────────────────────────────────────────────────────────


# def log(msg, level="INFO"):
#     timestamp = datetime.now().strftime("%H:%M:%S")
#     icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARN": "⚠️", "PROGRESS": "🔄"}
#     print(f"[{timestamp}] {icons.get(level, '')} {msg}")


# def print_progress_bar(current, total, prefix="", length=30):
#     filled = int(length * current / total)
#     bar = "█" * filled + "░" * (length - filled)
#     percent = 100 * current / total
#     print(f"\r  {prefix} |{bar}| {percent:.1f}% ({current}/{total})", end="", flush=True)
#     if current == total:
#         print()


# def print_banner():
#     existing = len(pd.read_csv(OUTPUT_PATH)) if os.path.exists(OUTPUT_PATH) else 0
#     print("\n" + "="*60)
#     print("   🔍 NAUKRI JOB SCRAPER")
#     print(f"   Job types          : {len(JOB_TYPES)}")
#     print(f"   Target per type    : {JOBS_PER_TYPE}")
#     print(f"   Total target       : {len(JOB_TYPES) * JOBS_PER_TYPE}")
#     print(f"   Existing jobs      : {existing}")
#     print("="*60 + "\n")


# def setup_driver():
#     log("Starting Chrome driver...")
#     options = Options()
#     options.add_argument("--headless=new")
#     options.add_argument("--disable-blink-features=AutomationControlled")
#     options.add_argument("--start-maximized")
#     options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
#     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
#     log("Chrome driver ready", "SUCCESS")
#     return driver


# def get_job_links(driver, job_type, page):
#     try:
#         if page == 1:
#             url = f"{BASE_URL}/{job_type}"
#         else:
#             url = f"{BASE_URL}/{job_type}-{page}"
#         driver.get(url)
#         time.sleep(random.uniform(3, 5))
#         soup = BeautifulSoup(driver.page_source, "html.parser")
#         links = []
#         cards = soup.find_all('div', class_=lambda x: x and 'cust-job-tuple' in str(x))
#         for card in cards:
#             a = card.find('a', class_='title')
#             if a and a.get('href'):
#                 links.append(a['href'])
#         return links
#     except Exception as e:
#         log(f"Error getting links: {e}", "ERROR")
#         return []


# def scrape_job(driver, url):
#     try:
#         driver.get(url)
#         time.sleep(random.uniform(3, 5))
#         soup = BeautifulSoup(driver.page_source, "html.parser")

#         # title
#         title = soup.find('h1', class_=lambda x: x and 'jd-header-title' in str(x))
#         title = title.text.strip() if title else None

#         # company
#         company = None
#         comp_el = soup.find(class_=lambda x: x and 'jd-header-comp-name' in str(x))
#         if comp_el:
#             a = comp_el.find('a')
#             company = a.text.strip() if a else comp_el.text.strip()[:50]

#         # full description
#         desc = None
#         desc_container = soup.find(class_=lambda x: x and 'job-desc-container' in str(x))
#         if desc_container:
#             desc = desc_container.get_text(separator=' ', strip=True)
#         else:
#             desc_el = soup.find(class_=lambda x: x and 'dang-inner-html' in str(x))
#             if desc_el:
#                 desc = desc_el.get_text(separator=' ', strip=True)

#         # skills
#         skills = []
#         skills_container = soup.find(class_=lambda x: x and 'key-skill' in str(x))
#         if skills_container:
#             skill_tags = skills_container.find_all('a')
#             skills = [s.text.strip() for s in skill_tags]

#         if not title or not desc:
#             return None

#         return {
#             "title": title,
#             "company": company,
#             "description": desc,
#             "skills": " | ".join(skills),
#             "url": url
#         }

#     except Exception as e:
#         log(f"Error scraping {url[-50:]}: {e}", "ERROR")
#         return None


# def scrape_all():
#     os.makedirs("data/raw", exist_ok=True)
#     print_banner()
#     driver = setup_driver()
#     all_jobs = []
#     start_time = datetime.now()

#     for type_idx, job_type in enumerate(JOB_TYPES, 1):
#         print(f"\n{'─'*60}")
#         log(f"[{type_idx}/{len(JOB_TYPES)}] Starting: {job_type}", "PROGRESS")
#         print(f"{'─'*60}")

#         # collect links
#         links = []
#         log("Collecting job links...")
#         for page in range(1, PAGES_PER_TYPE + 1):
#             page_links = get_job_links(driver, job_type, page)
#             links.extend(page_links)
#             log(f"Page {page} → {len(page_links)} links (total: {len(links)})")
#             if len(page_links) == 0:
#                 log("No more pages, moving on", "WARN")
#                 break
#             if len(links) >= JOBS_PER_TYPE:
#                 break
#             time.sleep(random.uniform(1, 2))

#         links = list(set(links))[:JOBS_PER_TYPE]
#         log(f"Total unique links: {len(links)}", "SUCCESS")

#         # scrape each job
#         type_jobs = []
#         log("Scraping job details...")
#         print()
#         for i, url in enumerate(links, 1):
#             print_progress_bar(i, len(links), prefix=f"  {job_type[:25]:<25}")
#             data = scrape_job(driver, url)

#             if data is None:
#                 try:
#                     driver.current_url
#                 except:
#                     log("Driver crashed — restarting...", "WARN")
#                     try:
#                         driver.quit()
#                     except:
#                         pass
#                     driver = setup_driver()
#             else:
#                 data["job_type"] = job_type
#                 type_jobs.append(data)
#                 all_jobs.append(data)

#         print()
#         log(f"Scraped {len(type_jobs)}/{len(links)} jobs", "SUCCESS")

#         # append to existing csv
#         if os.path.exists(OUTPUT_PATH):
#             existing_df = pd.read_csv(OUTPUT_PATH)
#             df = pd.concat([existing_df, pd.DataFrame(all_jobs)], ignore_index=True)
#             df = df.drop_duplicates(subset=["url"])
#         else:
#             df = pd.DataFrame(all_jobs)

#         df.to_csv(OUTPUT_PATH, index=False)
#         log(f"Saved → {OUTPUT_PATH} ({len(df)} total rows)", "SUCCESS")

#     driver.quit()

#     elapsed = datetime.now() - start_time
#     final_df = pd.read_csv(OUTPUT_PATH)
#     print(f"\n{'='*60}")
#     print(f"   🎉 SCRAPING COMPLETE")
#     print(f"   New jobs scraped   : {len(all_jobs)}")
#     print(f"   Total in CSV       : {len(final_df)}")
#     print(f"   Time taken         : {str(elapsed).split('.')[0]}")
#     print(f"   Saved to           : {OUTPUT_PATH}")
#     print(f"{'='*60}\n")


# if __name__ == "__main__":
#     scrape_all()


import time
import random
import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from datetime import datetime
from collections import defaultdict

# ── Config ──────────────────────────────────────────────
JOB_TYPES = [
    "machine-learning-jobs",
    "artificial-intelligence-jobs",
    "data-science-jobs",
    "data-analyst-jobs",
    "python-developer-jobs",
    "data-engineer-jobs",
    "deep-learning-jobs",
    "nlp-jobs",
    "computer-vision-jobs",
    "generative-ai-jobs",
]
JOBS_PER_TYPE = 40
PAGES_PER_TYPE = 8
MAX_PER_COMPANY = 3
OUTPUT_PATH = "data/raw/naukri_jobs.csv"
BASE_URL = "https://www.naukri.com"
# ────────────────────────────────────────────────────────


def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARN": "⚠️", "PROGRESS": "🔄"}
    print(f"[{timestamp}] {icons.get(level, '')} {msg}")


def print_progress_bar(current, total, prefix="", length=30):
    filled = int(length * current / total)
    bar = "█" * filled + "░" * (length - filled)
    percent = 100 * current / total
    print(f"\r  {prefix} |{bar}| {percent:.1f}% ({current}/{total})", end="", flush=True)
    if current == total:
        print()


def print_banner():
    existing = len(pd.read_csv(OUTPUT_PATH)) if os.path.exists(OUTPUT_PATH) else 0
    print("\n" + "="*60)
    print("   🔍 NAUKRI JOB SCRAPER")
    print(f"   Job types          : {len(JOB_TYPES)}")
    print(f"   Target per type    : {JOBS_PER_TYPE}")
    print(f"   Max per company    : {MAX_PER_COMPANY}")
    print(f"   Total target       : {len(JOB_TYPES) * JOBS_PER_TYPE}")
    print(f"   Existing jobs      : {existing}")
    print("="*60 + "\n")


def setup_driver():
    log("Starting Chrome driver...")
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    log("Chrome driver ready", "SUCCESS")
    return driver


def get_job_links(driver, job_type, page):
    try:
        url = f"{BASE_URL}/{job_type}" if page == 1 else f"{BASE_URL}/{job_type}-{page}"
        driver.get(url)
        time.sleep(random.uniform(3, 5))
        soup = BeautifulSoup(driver.page_source, "html.parser")
        links = []
        cards = soup.find_all('div', class_=lambda x: x and 'cust-job-tuple' in str(x))
        for card in cards:
            a = card.find('a', class_='title')
            if a and a.get('href'):
                links.append(a['href'])
        return links
    except Exception as e:
        log(f"Error getting links: {e}", "ERROR")
        return []


def scrape_job(driver, url):
    try:
        driver.get(url)
        time.sleep(random.uniform(3, 5))
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # title
        title = soup.find('h1', class_=lambda x: x and 'jd-header-title' in str(x))
        title = title.text.strip() if title else None

        # company
        company = None
        comp_el = soup.find(class_=lambda x: x and 'jd-header-comp-name' in str(x))
        if comp_el:
            a = comp_el.find('a')
            company = a.text.strip() if a else comp_el.text.strip()[:50]

        # full description
        desc = None
        desc_container = soup.find(class_=lambda x: x and 'job-desc-container' in str(x))
        if desc_container:
            desc = desc_container.get_text(separator=' ', strip=True)
        else:
            desc_el = soup.find(class_=lambda x: x and 'dang-inner-html' in str(x))
            if desc_el:
                desc = desc_el.get_text(separator=' ', strip=True)

        # skills
        skills = []
        skills_container = soup.find(class_=lambda x: x and 'key-skill' in str(x))
        if skills_container:
            skill_tags = skills_container.find_all('a')
            skills = [s.text.strip() for s in skill_tags]

        if not title or not desc:
            return None

        return {
            "title": title,
            "company": company if company else "Unknown",
            "description": desc,
            "skills": " | ".join(skills),
            "url": url
        }

    except Exception as e:
        log(f"Error scraping {url[-50:]}: {e}", "ERROR")
        return None


def scrape_all():
    os.makedirs("data/raw", exist_ok=True)
    print_banner()
    driver = setup_driver()
    all_jobs = []
    start_time = datetime.now()

    for type_idx, job_type in enumerate(JOB_TYPES, 1):
        print(f"\n{'─'*60}")
        log(f"[{type_idx}/{len(JOB_TYPES)}] Starting: {job_type}", "PROGRESS")
        print(f"{'─'*60}")

        # collect links
        links = []
        log("Collecting job links...")
        for page in range(1, PAGES_PER_TYPE + 1):
            page_links = get_job_links(driver, job_type, page)
            links.extend(page_links)
            log(f"Page {page} → {len(page_links)} links (total: {len(links)})")
            if len(page_links) == 0:
                log("No more pages, moving on", "WARN")
                break
            if len(links) >= JOBS_PER_TYPE * 2:  # collect 2x to allow for company filter
                break
            time.sleep(random.uniform(1, 2))

        links = list(set(links))
        log(f"Total unique links: {len(links)}", "SUCCESS")

        # scrape each job with company limit
        type_jobs = []
        company_count = defaultdict(int)  # track jobs per company
        log("Scraping job details...")
        print()

        for i, url in enumerate(links, 1):
            print_progress_bar(i, len(links), prefix=f"  {job_type[:25]:<25}")

            # stop if we have enough jobs for this type
            if len(type_jobs) >= JOBS_PER_TYPE:
                print()
                log(f"Reached {JOBS_PER_TYPE} jobs for {job_type}", "SUCCESS")
                break

            data = scrape_job(driver, url)

            if data is None:
                try:
                    driver.current_url
                except:
                    log("Driver crashed — restarting...", "WARN")
                    try:
                        driver.quit()
                    except:
                        pass
                    driver = setup_driver()
                continue

            # enforce max per company
            company_key = str(data["company"]).lower().strip()
            if company_count[company_key] >= MAX_PER_COMPANY:
                log(f"Skipping {data['company']} — max {MAX_PER_COMPANY} reached", "WARN")
                continue

            company_count[company_key] += 1
            data["job_type"] = job_type
            type_jobs.append(data)
            all_jobs.append(data)

        print()
        log(f"Scraped {len(type_jobs)} jobs for {job_type}", "SUCCESS")

        # append to existing csv
        if os.path.exists(OUTPUT_PATH):
            existing_df = pd.read_csv(OUTPUT_PATH)
            df = pd.concat([existing_df, pd.DataFrame(all_jobs)], ignore_index=True)
            df = df.drop_duplicates(subset=["url"])
            df = df.drop_duplicates(subset=["description"])
        else:
            df = pd.DataFrame(all_jobs)

        df.to_csv(OUTPUT_PATH, index=False)
        log(f"Saved → {OUTPUT_PATH} ({len(df)} total rows)", "SUCCESS")

    driver.quit()

    elapsed = datetime.now() - start_time
    final_df = pd.read_csv(OUTPUT_PATH)
    print(f"\n{'='*60}")
    print(f"   🎉 SCRAPING COMPLETE")
    print(f"   New jobs scraped   : {len(all_jobs)}")
    print(f"   Total in CSV       : {len(final_df)}")
    print(f"   Time taken         : {str(elapsed).split('.')[0]}")
    print(f"   Saved to           : {OUTPUT_PATH}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    scrape_all()