# /product_ai_service/main.py
import joblib
import json
import pandas as pd
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bs4 import BeautifulSoup, Tag
from typing import List

# --- DATA MODELS ---
class HTMLPayload(BaseModel):
    html: str

class Product(BaseModel):
    name: str | None = None
    price: str | None = None
    product_url: str | None = None
    image_url: str | None = None

# --- GLOBAL VARIABLES ---
# Load the model and features ONCE when the service starts
try:
    model = joblib.load("product_model.joblib")
    with open("model_features.json", 'r') as f:
        model_features = json.load(f)
    print("✅ Model and features loaded successfully.")
except FileNotFoundError:
    print("❌ Critical Error: Model or features file not found.")
    model = None
    model_features = []


# --- HELPER FUNCTIONS (Adapted from your notebook) ---
def extract_features_improved(tag: Tag) -> dict:
    text = tag.get_text(" ", strip=True)
    text_len = len(text)
    # Basic check to avoid division by zero
    word_count = len(text.split()) + 1
    char_count = text_len + 1
    
    features = {
        'text_length': text_len,
        'num_children': len(tag.find_all(recursive=False)),
        'has_price_symbol': 1 if re.search(r'[$€£¥]', text) else 0,
        'has_image': 1 if tag.find('img') else 0,
        'num_links': len(tag.find_all('a')),
        'avg_word_length': text_len / word_count,
        'digit_density': sum(c.isdigit() for c in text) / char_count,
        'is_div': 1 if tag.name == 'div' else 0,
        'is_li': 1 if tag.name == 'li' else 0,
        'is_article': 1 if tag.name == 'article' else 0,
    }
    return features

def extract_final_data(tag: Tag) -> Product:
    # This function is now more robust against missing elements
    name_tag = tag.find(['h1', 'h2', 'h3', 'h4'], class_=lambda c: c is None or 'title' in c) or tag.find('a', title=True)
    price_tag = tag.find(string=re.compile(r'[$€£¥\d]')) # Keep this as is for now
    link_tag = tag.find('a', href=True)
    img_tag = tag.find('img', src=True)

    # Make URL absolute if it's relative
    base_url = ""
    if link_tag and link_tag.get('href', '').startswith('/'):
        # This is a simplification; a more robust solution would pass the base domain
        pass # We'll handle this in the gateway if needed

    product_url = link_tag['href'] if link_tag else None
    image_url = img_tag.get('src') or img_tag.get('data-src') if img_tag else None

    return Product(
        name=name_tag.get_text(strip=True) if name_tag else "N/A",
        price=price_tag.strip() if price_tag and price_tag.strip() else "N/A",
        product_url=product_url,
        image_url=image_url
    )

# --- FASTAPI APP ---
app = FastAPI(
    title="Product AI Service",
    description="Uses a trained ML model to find and extract product info from HTML."
)

@app.post("/identify_products/", response_model=List[Product])
async def identify_products(payload: HTMLPayload):
    if not model or not model_features:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    soup = BeautifulSoup(payload.html, 'lxml')
    
    # ==================== NEW ACCURACY IMPROVEMENT LOGIC ====================
    # 1. First, try to find a main content area to reduce noise from headers/footers.
    #    Common IDs/tags for main content are 'main', 'content', 'primary'.
    search_area = soup.find('main') or soup.find(id='main') or soup.find(id='content') or soup.find(id='primary')
    # If we didn't find a specific area, fall back to the whole page body.
    if not search_area:
        search_area = soup.body if soup.body else soup
        
    print(f"🎯 Search area identified: <{search_area.name}>")

    # 2. Find all potential candidates within our focused search area.
    all_candidates = search_area.find_all(['div', 'li', 'article', 'section'])
    print(f"Found {len(all_candidates)} total candidates initially.")

    # 3. Explicitly filter out candidates that are inside headers, footers, or navs
    #    This is a safeguard in case the search_area logic wasn't specific enough.
    candidates = []
    for tag in all_candidates:
        if not tag.find_parent(['header', 'footer', 'nav', 'aside']):
            candidates.append(tag)
    
    print(f"Found {len(candidates)} candidates after filtering out header/footer/nav sections.")
    # ======================================================================

    if not candidates:
        return []

    # Continue the process with the much cleaner `candidates` list
    candidate_data = [extract_features_improved(tag) for tag in candidates]
    
    # Create DataFrame, ensuring it matches the model's expected features
    candidates_df = pd.DataFrame(candidate_data)
    for col in model_features:
        if col not in candidates_df.columns:
            candidates_df[col] = 0
    # Reorder columns to exactly match the model's training order
    candidates_df = candidates_df[model_features]

    # Predict which of the filtered candidates are actual products
    predictions = model.predict(candidates_df)
    
    # Get the actual BeautifulSoup tags for the items predicted as products (1)
    product_blocks = [tag for i, tag in enumerate(candidates) if predictions[i] == 1]
    
    # Apply a final confidence filter: a real product block should have both a link and an image
    final_blocks = [block for block in product_blocks if block.find('a', href=True) and block.find('img')]
    
    extracted_data = [extract_final_data(block) for block in final_blocks]
    print(f"✅ Identified {len(extracted_data)} final products after ML prediction and confidence filtering.")
    return extracted_data