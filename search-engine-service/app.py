# app.py - (Final, Corrected, Integrated, and OPTIMIZED Version)
import os
import time
import pandas as pd
from flask import Flask, request, jsonify
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from urllib.parse import urlparse
import joblib
# --- THE OPTIMIZATION: Import the tools for parallel processing ---
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)

# --- Configuration ---
API_KEY = os.environ.get("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.environ.get("GOOGLE_SEARCH_ENGINE_ID")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.60"))
PAGES_TO_SEARCH = int(os.environ.get("PAGES_TO_SEARCH", "1"))
# --- THE OPTIMIZATION: Define how many parallel scrapers to run ---
# This number can be tuned. 8 is a good starting point for a powerful machine.
MAX_SCRAPING_WORKERS = 8

CLASS_MAPPING = {0: 'B2B', 1: 'Producer', 2: 'Retailer', 3: 'Info/Blog'}
DOMAIN_BLACKLIST = [
    'reddit.com', 'facebook.com', 'instagram.com', 'twitter.com', 'youtube.com',
    'pinterest.com', 'linkedin.com', 'substack.com', 'blogspot.com'
]

# --- Model Loading ---
try:
    model = joblib.load('hybrid_supplier_model.joblib')
    model_columns = joblib.load('model_columns.joblib')
    print("✅ ML Models loaded successfully.")
except FileNotFoundError as e:
    print(f"❗️ CRITICAL ERROR: Could not load model file. Make sure .joblib files are in the /app directory. Error: {e}")
    raise e

# --- Helper Functions ---
def setup_driver():
    """Sets up a single instance of a headless Chrome browser."""
    options = ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')         # CRUCIAL for running in Docker
    options.add_argument('--disable-dev-shm-usage') # CRUCIAL for running in Docker
    options.add_argument('--log-level=3')
    try:
        # Inside Docker, the driver is at a system path, we don't need webdriver_manager
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
        return driver
    except Exception as e:
        print(f"❗️ CRITICAL ERROR: Could not start Selenium WebDriver: {e}")
        return None

def get_processed_query_from_nlp(raw_query: str) -> list:
    """Calls the nlp-service to get a list of expanded B2B search queries."""
    nlp_service_url = "http://nlp-service:8000/process-query"
    print(f"✨ Contacting NLP service at {nlp_service_url} for query: '{raw_query}'")
    try:
        response = requests.post(nlp_service_url, json={"query": raw_query}, timeout=60)
        response.raise_for_status()
        nlp_data = response.json()
        
        english_queries = nlp_data.get("search_data", {}).get("english", {}).get("b2b_search_queries", [])
        french_queries = nlp_data.get("search_data", {}).get("french", {}).get("b2b_search_queries", [])
        
        all_search_queries = english_queries + french_queries
        if not all_search_queries:
            all_search_queries = [raw_query]
        
        print(f"  -> NLP service returned {len(all_search_queries)} queries to search.")
        return all_search_queries

    except requests.exceptions.RequestException as e:
        print(f"❗️ CRITICAL: Could not connect to NLP service: {e}")
        print("  -> Fallback: Using the original, unprocessed query.")
        return [raw_query]

def search_google(query, start_index=1):
    """Searches Google for a single query and returns a list of URLs."""
    print(f"🔎 Searching Google for: '{query}'")
    try:
        service = build("customsearch", "v1", developerKey=API_KEY)
        res = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=10, start=start_index).execute()
        return [item['link'] for item in res.get('items', [])]
    except Exception as e:
        print(f"   -> ❗️ Google Search API Error: {e}")
        return []

def extract_features_for_prediction(driver, url):
    """Scrapes a single URL and extracts features for the ML model."""
    print(f"  -> Scraping {url}")
    try:
        driver.get(url)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string or ""
        h1 = soup.h1.string if soup.h1 else ""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        meta_desc_text = meta_desc['content'] if meta_desc else ""
        key_text = f"{title} {h1} {meta_desc_text}"
        body_text = soup.body.get_text().lower() if soup.body else ""
        features = {'key_text': key_text,'has_add_to_cart': 1 if 'add to cart' in body_text else 0,'has_login': 1 if 'login' in body_text or 'my account' in body_text else 0,'has_b2b_keywords': 1 if any(k in body_text for k in ['b2b', 'wholesale', 'distributor', 'trade', 'sourcing']) else 0,'num_links': len(soup.find_all('a')),'num_images': len(soup.find_all('img')),'text_to_html_ratio': len(body_text) / (len(html) + 1),'has_corporate_keywords': 1 if any(k in body_text for k in ['investor relations', 'careers', 'our company', 'investors']) else 0,'has_blog_keywords': 1 if any(k in body_text for k in ['byline', 'author:', 'published on', 'comments', 'leave a reply']) else 0,'url_contains_blog_path': 1 if any(p in url.lower() for p in ['/blog/', '/news/']) else 0}
        return pd.DataFrame([features])
    except Exception as e:
        print(f"     -> ❗️ Error processing page: {e}"); return None

# --- THE OPTIMIZATION: A dedicated function for processing one URL from start to finish ---
def process_single_url(url: str, model, model_columns) -> str | None:
    """
    This function contains all logic for scraping and classifying one URL.
    It returns the URL if it's a valid supplier, otherwise None.
    """
    if any(domain in url for domain in DOMAIN_BLACKLIST):
        print(f"  -> Skipping (Blacklist): {url}")
        return None

    driver = setup_driver()
    if not driver:
        return None
        
    try:
        features_df = extract_features_for_prediction(driver, url)
        if features_df is None:
            return None
        
        for col in model_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        features_df = features_df[model_columns]

        probabilities = model.predict_proba(features_df)[0]
        confidence = probabilities.max()
        prediction = probabilities.argmax()
        class_name = CLASS_MAPPING.get(prediction, "Unknown")
        print(f"     -> 🤖 [{class_name}] with {confidence:.2f} confidence for {url}")
        
        if prediction in [0, 1, 2] and confidence >= CONFIDENCE_THRESHOLD:
            print(f"        -> ✅ VALIDATED: {url}")
            return url
    except Exception as e:
        print(f"     -> ❗️ Error in worker for {url}: {e}")
    finally:
        if driver:
            driver.quit()
    
    return None

# --- API Endpoint (Now with Parallel Processing) ---
@app.route('/api/discover', methods=['POST'])
def discover_suppliers():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' key"}), 400
    
    raw_query = data['query']
    
    # Step 1: Call the NLP service to get the expanded queries
    all_search_queries = get_processed_query_from_nlp(raw_query)
    
    # Step 2: Loop through the new list of queries and gather all candidate URLs
    all_candidate_urls = set()
    for query in all_search_queries:
        for page in range(PAGES_TO_SEARCH):
            page_urls = search_google(query, start_index=(page * 10 + 1))
            if not page_urls: break
            all_candidate_urls.update(page_urls)
            time.sleep(1)

    unique_urls = list(all_candidate_urls)
    print(f"\nCollected {len(unique_urls)} unique candidates. Starting parallel processing...")
    
    # --- STEP 3: THE OPTIMIZATION - Process all URLs in parallel ---
    validated_urls = set()
    with ThreadPoolExecutor(max_workers=MAX_SCRAPING_WORKERS) as executor:
        future_to_url = {executor.submit(process_single_url, url, model, model_columns): url for url in unique_urls}
        
        for future in as_completed(future_to_url):
            result_url = future.result()
            if result_url: # If the function returned a valid URL
                validated_urls.add(result_url)
    
    print(f"\n--- Parallel processing complete ---")
    
    return jsonify({
        "original_query": raw_query,
        "expanded_queries_used": all_search_queries,
        "count": len(validated_urls),
        "supplier_urls": sorted(list(validated_urls))
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)