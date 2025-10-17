import time
import random
from typing import List, Tuple
import pandas as pd

from db_config import DatabaseManager
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager



class CompanyTracker:
    def __init__(self):
        self.db_manager = DatabaseManager(
            host="localhost",
            user="root", 
            password="",
            database="companies",
            port=3306
        )
        self.db_manager._ensure_initialized()
        self.current_changes: List[Dict[str, str]] = []
        self.added_pairs: List[Tuple[str, str]] = []
        print("✅ CompanyTracker initialized successfully with MySQL!")

def _setup_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Safari/537.36")

    drv = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    try:
        drv.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
        )
    except Exception:
        pass
    return drv


def _fetch_one(drv: webdriver.Chrome, cin: str) -> dict:
    url = f"https://www.zaubacorp.com/companysearchresults/{cin}"
    try:
        drv.get(url)
        WebDriverWait(drv, 12).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        rows = drv.find_elements(By.CSS_SELECTOR, "table tbody tr")
        if rows:
            tds = rows[0].find_elements(By.TAG_NAME, "td")
            if len(tds) >= 3:
                matched = tds[0].text.strip()
                name = tds[1].text.strip()
                addr = tds[2].text.strip()
                try:
                    detail = tds[1].find_element(By.TAG_NAME, "a").get_attribute("href")
                except Exception:
                    detail = url
                return {
                    "status": "success",
                    "matched_CIN": matched,
                    "companyName": name,
                    "address": addr,
                    "detailURL": detail,
                    "searchURL": url,
                }
        return {"status": "no_results", "error": "No company found", "searchURL": url}
    except Exception as e:
        return {"status": "error", "error": str(e), "searchURL": url}


def enrich_added(added_pairs: List[Tuple[str, str]],
                 output_excel_path: str = "added_enriched.xlsx",
                 headless: bool = True,
                 min_delay: float = 2.5,
                 max_delay: float = 5.0) -> pd.DataFrame:
  
    if not added_pairs:
        return pd.DataFrame(columns=["CIN","CompanyName","status","matched_CIN","companyName","address","detailURL","searchURL","error"])

    drv = _setup_driver(headless=headless)
    out = []
    try:
        for cin, cname in added_pairs:
            base = {"CIN": cin or "", "CompanyName": cname or ""}
            data = _fetch_one(drv, cin or "")
            out.append({**base, **data})
            time.sleep(random.uniform(min_delay, max_delay))
    finally:
        drv.quit()

    df = pd.DataFrame(out)
    wanted = ["CIN","CompanyName","status","matched_CIN","companyName","address","detailURL","searchURL","error"]
    for col in wanted:
        if col not in df.columns:
            df[col] = ""
    df = df[wanted]
    df.to_excel(output_excel_path, index=False)
    return df
