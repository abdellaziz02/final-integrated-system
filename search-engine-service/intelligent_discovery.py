# intelligent_discovery.py (Production-Ready Version)
import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from urllib.parse import urlparse
import joblib

# --- Configuration ---
API_KEY = 'AIzaSyDzSBrrRb6MDtQWZ2pWfholA9Y7NSbaFkc'
SEARCH_ENGINE_ID = "c353e359620404acc"
SEARCH_QUERIES = [
    "wholesale cherry tomatoes",
    "bulk cherry tomatoes supplier",
    "cherry tomato distributor USA",
    "b2b cherry tomato producer Italy"
]
OUTPUT_URL_FILE = "final_supplier_urls_cherry_tomato.txt"
CLASS_MAPPING = {0: 'B2B', 1: 'Producer', 2: 'Retailer', 3: 'Info/Blog'}

# --- UPGRADE: Blacklist of domains to automatically reject ---
DOMAIN_BLACKLIST = [
    'reddit.com', 'facebook.com', 'instagram.com', 'twitter.com', 'youtube.com',
    'pinterest.com', 'linkedin.com', 'substack.com', 'blogspot.com'
]

# (Helper functions setup_driver, search_google, extract_features_for_prediction remain the same)
def setup_driver():
    options = webdriver.ChromeOptions(); options.add_argument('--headless'); options.add_argument('--log-level=3')
    options.page_load_strategy = 'normal'
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(30)
    return driver
def search_google(query):
    print(f"\n🔎 Searching Google for: '{query}'")
    try:
        service = build("customsearch", "v1", developerKey=API_KEY)
        res = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=10).execute()
        return [item['link'] for item in res.get('items', [])]
    except Exception as e:
        print(f"   -> ❗️ Google Search API Error: {e}")
        return []
def extract_features_for_prediction(driver, url):
    print(f"  -> Processing {url}")
    try:
        driver.get(url); time.sleep(1)
        html = driver.page_source; soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string if soup.title else ""; h1 = soup.h1.string if soup.h1 else ""
        meta_desc = soup.find('meta', attrs={'name': 'description'}); meta_desc_text = meta_desc['content'] if meta_desc else ""
        key_text = f"{title} {h1} {meta_desc_text}"
        body_text = soup.body.get_text().lower() if soup.body else ""
        features = {'key_text': key_text,'has_add_to_cart': 1 if 'add to cart' in body_text else 0,'has_login': 1 if 'login' in body_text or 'my account' in body_text else 0,'has_b2b_keywords': 1 if any(k in body_text for k in ['b2b', 'wholesale', 'distributor', 'trade', 'sourcing']) else 0,'num_links': len(soup.find_all('a')),'num_images': len(soup.find_all('img')),'text_to_html_ratio': len(body_text) / (len(html) + 1),'has_corporate_keywords': 1 if any(k in body_text for k in ['investor relations', 'careers', 'our company', 'investors']) else 0,'has_blog_keywords': 1 if any(k in body_text for k in ['byline', 'author:', 'published on', 'comments', 'leave a reply']) else 0,'url_contains_blog_path': 1 if any(p in url.lower() for p in ['/blog/', '/news/']) else 0,}
        return pd.DataFrame([features])
    except Exception as e:
        print(f"     -> ❗️ Error processing page: {e}"); return None

# --- Main Execution Block ---
if __name__ == "__main__":
    model = joblib.load('hybrid_supplier_model.joblib'); model_columns = joblib.load('model_columns.joblib')
    print("✅ Hybrid model and columns loaded successfully.")
    
    validated_supplier_urls = set()
    
    for query in SEARCH_QUERIES:
        driver = setup_driver()
        print(f"\n--- Starting new query with fresh browser ---")
        for url in search_google(query):
            
            # --- UPGRADE: Applying the Hard-Coded Guardrail ---
            parsed_url = urlparse(url)
            if any(domain in parsed_url.netloc for domain in DOMAIN_BLACKLIST):
                print(f"  -> Rejecting (on blacklist): {url}")
                continue # Skip this URL entirely
            
            features_df = extract_features_for_prediction(driver, url)
            if features_df is None: continue
                
            prediction = model.predict(features_df[model_columns])[0]
            class_name = CLASS_MAPPING[prediction]
            print(f"     -> 🤖 Model classified as: [{class_name}]")
            if prediction in [0, 1, 2]:
                print(f"        -> ✅ VALIDATED as a potential supplier.")
                validated_supplier_urls.add(url)
        driver.quit()
    
    print("\n\n--- DISCOVERY COMPLETE ---")
    if validated_supplier_urls:
        print(f"Found {len(validated_supplier_urls)} potential supplier URLs for cherry tomatoes.")
        with open(OUTPUT_URL_FILE, 'w', encoding='utf-8') as f:
            for url in sorted(list(validated_supplier_urls)):
                f.write(url + '\n')
        print(f"✅ Clean list of URLs saved to: {OUTPUT_URL_FILE}")
    else:
        print("No potential suppliers were found for the given queries.")