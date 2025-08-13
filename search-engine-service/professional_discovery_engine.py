# professional_discovery_engine.py
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. Configuration ---
API_KEY = 'AIzaSyDzSBrrRb6MDtQWZ2pWfholA9Y7NSbaFkc'
SEARCH_ENGINE_ID = "c353e359620404acc"
SEARCH_QUERIES = [
    "wholesale cherry tomatoes",
    "bulk cherry tomatoes supplier",
    "b2b organic tofu manufacturer Europe",
    "oat milk bulk supplier",
    "private label kombucha brewer",
    "plant-based meat producer Germany"
]
OUTPUT_CSV_FILE = "final_supplier_results.csv"
CLASS_MAPPING = {0: 'B2B', 1: 'Producer', 2: 'Retailer', 3: 'Info/Blog'}
DOMAIN_BLACKLIST = ['reddit.com', 'facebook.com', 'instagram.com', 'twitter.com', 'youtube.com', 'pinterest.com', 'linkedin.com']

# --- ENHANCEMENT 1: The Topicality Filter ---
# Keywords that MUST appear on a page for it to be considered relevant.
TOPIC_KEYWORDS = [
    'food', 'ingredient', 'beverage', 'produce', 'organic', 'farm', 'harvest',
    'culinary', 'gourmet', 'kitchen', 'recipe', 'nutrition', 'agriculture',
    'tomato', 'strawberry', 'oat', 'milk', 'cheese', 'pasta', 'tofu', 'meat'
]

# --- 2. Setup & Helper Functions ---
def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless'); options.add_argument('--log-level=3')
    # --- ENHANCEMENT: Faster page loads by disabling images/css ---
    options.add_argument('--blink-settings=imagesEnabled=false')
    prefs = {'profile.managed_default_content_settings.images': 2}
    options.add_experimental_option('prefs', prefs)
    return webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

def search_google(query):
    # (Unchanged)
    print(f"\n🔎 Searching Google for: '{query}'")
    try:
        service = build("customsearch", "v1", developerKey=API_KEY)
        res = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=10).execute()
        return [item['link'] for item in res.get('items', [])]
    except Exception as e:
        print(f"   -> ❗️ Google Search API Error: {e}")
        return []

def process_url(url, model, model_columns):
    """
    This function contains all the logic for processing a SINGLE URL.
    It will be run in parallel for multiple URLs.
    """
    # Guardrail 1: Blacklist
    parsed_url = urlparse(url)
    if any(domain in parsed_url.netloc for domain in DOMAIN_BLACKLIST):
        print(f"  -> Rejecting (Blacklist): {url}")
        return None

    driver = setup_driver()
    try:
        # Step 1: Scrape page and extract features
        driver.get(url)
        time.sleep(1)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        
        body_text = soup.body.get_text().lower() if soup.body else ""

        # Guardrail 2: Topicality Filter
        if not any(keyword in body_text for keyword in TOPIC_KEYWORDS):
            print(f"  -> Rejecting (Not on topic): {url}")
            return None
        
        # Step 2: Extract features for the model
        title = soup.title.string or ""; h1 = soup.h1.string or ""
        meta_desc = soup.find('meta', attrs={'name': 'description'}); meta_desc_text = meta_desc['content'] if meta_desc else ""
        key_text = f"{title} {h1} {meta_desc_text}"
        
        features = {'key_text': key_text,'has_add_to_cart': 1 if 'add to cart' in body_text else 0,'has_login': 1 if 'login' in body_text or 'my account' in body_text else 0,'has_b2b_keywords': 1 if any(k in body_text for k in ['b2b', 'wholesale', 'distributor', 'trade', 'sourcing']) else 0,'num_links': len(soup.find_all('a')),'num_images': len(soup.find_all('img')),'text_to_html_ratio': len(body_text) / (len(html) + 1),'has_corporate_keywords': 1 if any(k in body_text for k in ['investor relations', 'careers', 'our company', 'investors']) else 0,'has_blog_keywords': 1 if any(k in body_text for k in ['byline', 'author:', 'published on', 'comments', 'leave a reply']) else 0,'url_contains_blog_path': 1 if any(p in url.lower() for p in ['/blog/', '/news/']) else 0}
        features_df = pd.DataFrame([features])

        # Step 3: Classify with the model
        prediction = model.predict(features_df[model_columns])[0]
        class_name = CLASS_MAPPING[prediction]
        print(f"  -> Processing {url} ... Classified as: [{class_name}]")

        # Step 4: Return result if it's a valid supplier type
        if prediction in [0, 1, 2]:
            return {'url': url, 'predicted_category': class_name}
        
    except Exception as e:
        print(f"     -> ❗️ Error processing {url}: {e}")
    finally:
        driver.quit()
    
    return None

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    model = joblib.load('hybrid_supplier_model.joblib')
    model_columns = joblib.load('model_columns.joblib')
    print("✅ Hybrid model and columns loaded successfully.")
    
    all_candidate_urls = []
    for query in SEARCH_QUERIES:
        all_candidate_urls.extend(search_google(query))
    
    # Use a set to get unique URLs
    unique_urls = list(dict.fromkeys(all_candidate_urls))
    print(f"\nCollected {len(unique_urls)} unique candidates to process.")
    
    final_results = []
    
    # --- ENHANCEMENT 2: Parallel Processing ---
    # We create a pool of worker threads to process URLs simultaneously.
    # Adjust max_workers based on your computer's power (4-8 is a good start).
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all URL processing tasks to the pool
        future_to_url = {executor.submit(process_url, url, model, model_columns): url for url in unique_urls}
        
        for future in as_completed(future_to_url):
            result = future.result()
            if result: # If the function returned a valid supplier
                final_results.append(result)
                
    print("\n\n--- DISCOVERY COMPLETE ---")
    
    if final_results:
        # --- ENHANCEMENT 3: Structured Output ---
        results_df = pd.DataFrame(final_results)
        results_df.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"Found {len(results_df)} potential suppliers.")
        print(f"✅ Results saved to: {OUTPUT_CSV_FILE}")
        print("\n--- Top Results ---")
        print(results_df.head())
    else:
        print("No potential suppliers were found for the given queries.")