import json
import time
import re
import random
import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from datetime import datetime

# ── Config ──────────────────────────────────────────────
JOB_TYPES = [
    
    "python-jobs",
    "big-data-jobs",
    "machine-learning-jobs",
    "artificial-intelligence-jobs",
    "AI-Engeneering-jobs",
    "research-analyst-jobs"
]
JOBS_PER_TYPE = 50
PAGES_PER_TYPE = 10
OUTPUT_PATH = "data/raw/internshala_jobs.csv"
# ────────────────────────────────────────────────────────


def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARN": "⚠️", "PROGRESS": "🔄"}
    print(f"[{timestamp}] {icons.get(level, '')} {msg}")


def print_banner():
    existing = 0
    if os.path.exists(OUTPUT_PATH):
        existing = len(pd.read_csv(OUTPUT_PATH))
    print("\n" + "="*60)
    print("   🔍 INTERNSHALA JOB SCRAPER — APPEND MODE")
    print(f"   Existing jobs      : {existing}")
    print(f"   New job types      : {len(JOB_TYPES)}")
    print(f"   Target new jobs    : {len(JOB_TYPES) * JOBS_PER_TYPE}")
    print(f"   Expected total     : {existing + len(JOB_TYPES) * JOBS_PER_TYPE}+")
    print("="*60 + "\n")


def print_progress_bar(current, total, prefix="", length=30):
    filled = int(length * current / total)
    bar = "█" * filled + "░" * (length - filled)
    percent = 100 * current / total
    print(f"\r  {prefix} |{bar}| {percent:.1f}% ({current}/{total})", end="", flush=True)
    if current == total:
        print()


def setup_driver():
    log("Starting Chrome driver...")
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    log("Chrome driver ready", "SUCCESS")
    return driver


def get_job_links(driver, job_type, page):
    url = f"https://internshala.com/jobs/{job_type}/page-{page}/"
    driver.get(url)
    time.sleep(random.uniform(2, 4))
    soup = BeautifulSoup(driver.page_source, "lxml")
    links = []
    for a in soup.select("a.job-title-href"):
        href = a.get("href")
        if href:
            if href.startswith("/"):
                href = "https://internshala.com" + href
            links.append(href)
    return links


def scrape_job(driver, url):
    try:
        if url.startswith("/"):
            url = "https://internshala.com" + url
        if not url.startswith("http"):
            log(f"Invalid URL skipped: {url}", "WARN")
            return None

        driver.get(url)
        time.sleep(random.uniform(3, 5))
        soup = BeautifulSoup(driver.page_source, "lxml")

        # Title
        title = soup.find("h1")
        title = title.text.strip() if title else None

        # Company
        company = None
        internship_details = soup.find("div", class_="internship_details")
        if internship_details:
            for h2 in reversed(internship_details.find_all("h2")):
                text = h2.text.strip()
                if "about" in text.lower() and "job" not in text.lower():
                    company = re.sub(r"^About\s*", "", text).strip()
                    break

        # Description
        description = None
        if internship_details:
            about_section = internship_details.find("h2", class_="about_heading")
            if about_section:
                tc = about_section.find_next_sibling("div", class_="text-container")
                if tc:
                    description = tc.get_text(separator=" ", strip=True)

        # Skills
        skills = []
        skills_heading = soup.find("h3", class_="skills_heading")
        if skills_heading:
            skills_container = skills_heading.find_next_sibling("div", class_="round_tabs_container")
            if skills_container:
                skills = [s.text.strip() for s in skills_container.find_all("span", class_="round_tabs")]

        # Other requirements
        other_requirements = None
        additional = soup.find("div", class_="additional_detail")
        if additional:
            other_requirements = additional.get_text(separator=" ", strip=True)

        return {
            "title": title,
            "company": company,
            "skills": " | ".join(skills),
            "description": description,
            "other_requirements": other_requirements,
            "job_type": None,  # will be set in scrape_all
            "url": url
        }

    except Exception as e:
        log(f"Error scraping {url[-40:]}: {e}", "ERROR")
        return None


def scrape_all():
    os.makedirs("data/raw", exist_ok=True)
    print_banner()
    driver = setup_driver()
    all_jobs = []
    total_types = len(JOB_TYPES)
    start_time = datetime.now()

    for type_idx, job_type in enumerate(JOB_TYPES, 1):
        print(f"\n{'─'*60}")
        log(f"[{type_idx}/{total_types}] Starting: {job_type}", "PROGRESS")
        print(f"{'─'*60}")

        # Collect links
        links = []
        log("Collecting job links...")
        for page in range(1, PAGES_PER_TYPE + 1):
            try:
                page_links = get_job_links(driver, job_type, page)
                links.extend(page_links)
                log(f"Page {page} → {len(page_links)} links found (total: {len(links)})")
                if len(links) >= JOBS_PER_TYPE:
                    break
                if len(page_links) == 0:
                    log("No more pages found, moving on", "WARN")
                    break
                time.sleep(random.uniform(1, 2))
            except Exception as e:
                log(f"Error getting links page {page}: {e}", "ERROR")
                try:
                    driver.current_url
                except:
                    log("Driver crashed — restarting...", "WARN")
                    try:
                        driver.quit()
                    except:
                        pass
                    driver = setup_driver()

        links = list(set(links))[:JOBS_PER_TYPE]
        log(f"Total unique links: {len(links)}", "SUCCESS")

        # Scrape each job
        type_jobs = []
        log("Scraping job details...")
        print()
        for i, url in enumerate(links, 1):
            print_progress_bar(i, len(links), prefix=f"  {job_type[:25]:<25}")
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
            else:
                data["job_type"] = job_type
                type_jobs.append(data)
                all_jobs.append(data)

        print()
        log(f"Scraped {len(type_jobs)}/{len(links)} jobs successfully", "SUCCESS")

        # Append to existing CSV
        if os.path.exists(OUTPUT_PATH):
            existing_df = pd.read_csv(OUTPUT_PATH)
            new_df = pd.DataFrame(all_jobs)
            df = pd.concat([existing_df, new_df], ignore_index=True)
            df = df.drop_duplicates(subset=["url"])
        else:
            df = pd.DataFrame(all_jobs)

        df.to_csv(OUTPUT_PATH, index=False)
        log(f"Auto-saved → {OUTPUT_PATH} ({len(df)} total rows)", "SUCCESS")

    driver.quit()

    # Final summary
    final_df = pd.read_csv(OUTPUT_PATH)
    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"   🎉 SCRAPING COMPLETE")
    print(f"   New jobs scraped   : {len(all_jobs)}")
    print(f"   Total in CSV       : {len(final_df)}")
    print(f"   Time taken         : {str(elapsed).split('.')[0]}")
    print(f"   Saved to           : {OUTPUT_PATH}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    scrape_all()

# import json
# import time
# import re
# import random
# import pandas as pd
# import os
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from webdriver_manager.chrome import ChromeDriverManager
# from bs4 import BeautifulSoup
# from datetime import datetime

# # ── Config ──────────────────────────────────────────────
# JOB_TYPES = [
#     "machine-learning-jobs",
#     "artificial-intelligence-jobs",
#     "data-science-jobs",
#     "python-developer-jobs",
#     "data-analyst-jobs"
# ]
# JOBS_PER_TYPE = 50
# PAGES_PER_TYPE = 10
# OUTPUT_PATH = "data/raw/internshala_jobs.csv"
# # ────────────────────────────────────────────────────────


# def log(msg, level="INFO"):
#     timestamp = datetime.now().strftime("%H:%M:%S")
#     icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARN": "⚠️", "PROGRESS": "🔄"}
#     print(f"[{timestamp}] {icons.get(level, '')} {msg}")


# def print_banner():
#     print("\n" + "="*60)
#     print("   🔍 INTERNSHALA JOB SCRAPER")
#     print(f"   Target: {len(JOB_TYPES)} job types × {JOBS_PER_TYPE} jobs = {len(JOB_TYPES)*JOBS_PER_TYPE} total JDs")
#     print("="*60 + "\n")


# def print_progress_bar(current, total, prefix="", length=30):
#     filled = int(length * current / total)
#     bar = "█" * filled + "░" * (length - filled)
#     percent = 100 * current / total
#     print(f"\r  {prefix} |{bar}| {percent:.1f}% ({current}/{total})", end="", flush=True)
#     if current == total:
#         print()


# def setup_driver():
#     log("Starting Chrome driver...")
#     options = Options()
#     options.add_argument("--headless=new")
#     options.add_argument("--disable-blink-features=AutomationControlled")
#     options.add_argument("--start-maximized")
#     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
#     log("Chrome driver ready", "SUCCESS")
#     return driver


# def get_job_links(driver, job_type, page):
#     url = f"https://internshala.com/jobs/{job_type}/page-{page}/"
#     driver.get(url)
#     time.sleep(random.uniform(2, 4))
#     soup = BeautifulSoup(driver.page_source, "lxml")
#     links = []
#     for a in soup.select("a.job-title-href"):
#         href = a.get("href")
#         if href:
#             # Fix relative URLs
#             if href.startswith("/"):
#                 href = "https://internshala.com" + href
#             links.append(href)
#     return links
# def scrape_job(driver, url):
#     try:
#         # Fix relative URLs
#         if url.startswith("/"):
#             url = "https://internshala.com" + url
            
#         if not url.startswith("http"):
#             log(f"Invalid URL skipped: {url}", "WARN")
#             return None
            
#         driver.get(url)
#         time.sleep(random.uniform(3, 5))
#         soup = BeautifulSoup(driver.page_source, "lxml")

#         title = soup.find("h1")
#         title = title.text.strip() if title else None

#         company = None
#         internship_details = soup.find("div", class_="internship_details")
#         if internship_details:
#             for h2 in reversed(internship_details.find_all("h2")):
#                 text = h2.text.strip()
#                 if "about" in text.lower() and "job" not in text.lower():
#                     company = re.sub(r"^About\s*", "", text).strip()
#                     break

#         description = None
#         if internship_details:
#             about_section = internship_details.find("h2", class_="about_heading")
#             if about_section:
#                 tc = about_section.find_next_sibling("div", class_="text-container")
#                 if tc:
#                     description = tc.get_text(separator=" ", strip=True)

#         skills = []
#         skills_heading = soup.find("h3", class_="skills_heading")
#         if skills_heading:
#             skills_container = skills_heading.find_next_sibling("div", class_="round_tabs_container")
#             if skills_container:
#                 skills = [s.text.strip() for s in skills_container.find_all("span", class_="round_tabs")]

#         other_requirements = None
#         additional = soup.find("div", class_="additional_detail")
#         if additional:
#             other_requirements = additional.get_text(separator=" ", strip=True)

#         return {
#             "title": title,
#             "company": company,
#             "skills": " | ".join(skills),
#             "description": description,
#             "other_requirements": other_requirements,
#             "url": url
#         }

#     except Exception as e:
#         log(f"Error scraping {url[-40:]}: {e}", "ERROR")
#         return None


# def scrape_all():
#     os.makedirs("data/raw", exist_ok=True)
#     print_banner()
#     driver = setup_driver()
#     all_jobs = []
#     total_types = len(JOB_TYPES)
#     start_time = datetime.now()

#     for type_idx, job_type in enumerate(JOB_TYPES, 1):
#         print(f"\n{'─'*60}")
#         log(f"[{type_idx}/{total_types}] Starting: {job_type}", "PROGRESS")
#         print(f"{'─'*60}")

#         # Collect links
#         links = []
#         log(f"Collecting job links...")
#         for page in range(1, PAGES_PER_TYPE + 1):
#             try:
#                 page_links = get_job_links(driver, job_type, page)
#                 links.extend(page_links)
#                 log(f"Page {page} → {len(page_links)} links found (total: {len(links)})")
#                 if len(links) >= JOBS_PER_TYPE:
#                     break
#                 time.sleep(random.uniform(1, 2))
#             except Exception as e:
#                 log(f"Error getting links page {page}: {e}", "ERROR")
#                 try:
#                     driver.current_url
#                 except:
#                     log("Driver crashed while getting links — restarting...", "WARN")
#                     try:
#                         driver.quit()
#                     except:
#                         pass
#                     driver = setup_driver()

#         links = list(set(links))[:JOBS_PER_TYPE]
#         log(f"Total unique links: {len(links)}", "SUCCESS")

#         # Scrape each job
#         type_jobs = []
#         log(f"Scraping job details...")
#         print()
#         for i, url in enumerate(links, 1):
#             print_progress_bar(i, len(links), prefix=f"  {job_type[:25]:<25}")
#             data = scrape_job(driver, url)

#             # If driver crashed, restart it
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
#                 type_jobs.append(data)
#                 all_jobs.append(data)

#         print()
#         log(f"Scraped {len(type_jobs)}/{len(links)} jobs successfully", "SUCCESS")

#         # Save after each job type
#         df = pd.DataFrame(all_jobs)
#         df.to_csv(OUTPUT_PATH, index=False)
#         log(f"Auto-saved → {OUTPUT_PATH} ({len(all_jobs)} total rows)", "SUCCESS")

#     driver.quit()

#     # Final summary
#     elapsed = datetime.now() - start_time
#     print(f"\n{'='*60}")
#     print(f"   🎉 SCRAPING COMPLETE")
#     print(f"   Total jobs scraped : {len(all_jobs)}")
#     print(f"   Time taken         : {str(elapsed).split('.')[0]}")
#     print(f"   Saved to           : {OUTPUT_PATH}")
#     print(f"{'='*60}\n")


# if __name__ == "__main__":
#     scrape_all()