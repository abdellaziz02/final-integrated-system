# feature_extractor.py (Final Robust Version)
import re
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

TRAINING_FILES = ['supplier_training_multiclass.txt', 'verified_data_batch_1.txt',  'verified_data_batch_2.txt',
    'verified_data_batch_3.txt',
    'verified_data_batch_4.txt']
OUTPUT_CSV_FILE = 'supplier_engineered_features.csv'

def setup_driver():
    options = webdriver.ChromeOptions(); options.add_argument('--headless'); options.add_argument('--log-level=3')
    # --- UPGRADE: Set a longer page load timeout for the entire driver ---
    options.page_load_strategy = 'normal'
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(30) # Set a 30-second timeout for page loads
    return driver

def parse_url_file(filename):
    urls = [];
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            current_class = -1
            for line in f:
                if line.startswith("# Class"): current_class = int(re.search(r'\d+', line).group())
                if line.startswith('http') and current_class != -1: urls.append({'url': line.strip(), 'label': current_class})
    except FileNotFoundError: return []
    return urls

def extract_features(driver, url, retries=1):
    print(f"  Extracting features from: {url}")
    try:
        driver.get(url)
        # Instead of a fixed sleep, we can use a small one just to be safe
        time.sleep(1) 
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        
        title = soup.title.string if soup.title else ""
        h1 = soup.h1.string if soup.h1 else ""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        meta_desc_text = meta_desc['content'] if meta_desc else ""
        key_text = f"{title} {h1} {meta_desc_text}"
        body_text = soup.body.get_text().lower() if soup.body else ""
        
        features = {
            'has_add_to_cart': 1 if 'add to cart' in body_text else 0,
            'has_login': 1 if 'login' in body_text or 'my account' in body_text else 0,
            'has_b2b_keywords': 1 if any(k in body_text for k in ['b2b', 'wholesale', 'distributor', 'trade', 'sourcing']) else 0,
            'num_links': len(soup.find_all('a')), 'num_images': len(soup.find_all('img')),
            'text_to_html_ratio': len(body_text) / (len(html) + 1),
            'has_corporate_keywords': 1 if any(k in body_text for k in ['investor relations', 'careers', 'our company', 'investors']) else 0,
            'has_blog_keywords': 1 if any(k in body_text for k in ['byline', 'author:', 'published on', 'comments', 'leave a reply']) else 0,
            'url_contains_blog_path': 1 if any(p in url.lower() for p in ['/blog/', '/news/']) else 0,
        }
        return key_text, features
    except Exception as e:
        print(f"    -> Attempt failed: {e}")
        if retries > 0:
            print("    -> Retrying...")
            return extract_features(driver, url, retries - 1)
        return None, None

# --- Main Script ---
all_urls = []; [all_urls.extend(parse_url_file(f)) for f in TRAINING_FILES]
df_urls = pd.DataFrame(all_urls).drop_duplicates(subset='url', keep='last').reset_index(drop=True)
driver = setup_driver()
all_features_data = []

for index, row in df_urls.iterrows():
    key_text, structural_features = extract_features(driver, row['url'])
    if key_text and structural_features:
        record = {'url': row['url'], 'label': row['label'], 'key_text': key_text}
        record.update(structural_features)
        all_features_data.append(record)

driver.quit()
df_features = pd.DataFrame(all_features_data)
df_features.to_csv(OUTPUT_CSV_FILE, index=False)
print(f"\nFeature extraction complete. Saved to {OUTPUT_CSV_FILE}")