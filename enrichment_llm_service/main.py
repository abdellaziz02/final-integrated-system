from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import os
from groq import Groq

app = FastAPI()

# --- Configuration and Client Setup ---
# Your code uses GROQ_API_KEY1, which is fine, we just need to ensure the variable is set via docker-compose.
api_key = os.environ.get("GROQ_API_KEY1") 
client = Groq(api_key=api_key) if api_key else None

if not api_key:
    # Use the correct variable name in the warning message
    print("❌ WARNING: GROQ_API_KEY1 environment variable not set. The /enrich/ endpoint will fail.")
else:
    print("✅ GROQ_API_KEY1 found and initializing client.")

# --- Pydantic Models ---

# Input model for a single product from the product_ai_service
class RawProductData(BaseModel):
    # We make all these fields Optional because the upstream scraper may not find them
    name: Optional[str] = None
    price: Optional[str] = None
    product_url: Optional[str] = None
    image_url: Optional[str] = None

# Input model for the entire request (a list of products)
class EnrichmentRequest(BaseModel):
    products: List[RawProductData]

# Output model for a single enriched product
class EnrichedProductData(BaseModel):
    cleaned_name: str
    price_as_float: Optional[float] = None
    currency: Optional[str] = None
    image_url: Optional[str] = None
    # We add original_url back to the response for context
    original_url: Optional[str] = None 

# --- API Endpoint ---

# The endpoint now expects an EnrichmentRequest (which contains a list of products)
# And the response is a list of EnrichedProductData
@app.post("/enrich/", response_model=List[EnrichedProductData])
async def enrich_product_data(request: EnrichmentRequest):
    if not client:
        raise HTTPException(status_code=500, detail="LLM Client not configured.")

    enriched_products = []
    
    # We iterate over the list of products in the request body
    for product_data in request.products:
        # We only try to enrich if the product has a name and price
        if not product_data.name or not product_data.price:
            print(f"❌ Skipping product due to missing name or price: {product_data.product_url}")
            continue

        system_prompt = """
        You are an expert data cleaning bot. Your task is to take raw, messy product
        data and convert it into a clean, structured JSON object.
        - The `cleaned_name` should be a concise, human-readable name for the product.
        - The `price_as_float` must be a number (float), with no currency symbols or commas.
        - The `currency` should be the 3-letter currency code (e.g., USD, EUR, GBP).
        - ONLY output a valid JSON object. Do not include any other text or explanations.
        """

        user_prompt_data = {
            "name": product_data.name,
            "price": product_data.price
        }
        # We use the json.dumps() of the dictionary for the prompt
        user_prompt = f"Please process the following data: {json.dumps(user_prompt_data)}"

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model="llama3-8b-8192", # Llama 3 8B is excellent for this specific task
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            response_content = chat_completion.choices[0].message.content
            print("🧠 Raw LLM response:", repr(response_content))

            # Parse the JSON response from the LLM
            enriched_data = json.loads(response_content)
            
            # We add back context from the original raw data
            enriched_data['image_url'] = product_data.image_url
            enriched_data['original_url'] = product_data.product_url

            # Use Pydantic to validate the output before adding to the list
            enriched_products.append(EnrichedProductData(**enriched_data))

        except Exception as e:
            print(f"❌ An error occurred during LLM enrichment for product '{product_data.name}': {e}")
            # Do not raise an exception, just skip this product and continue
    
    # Return the list of successfully enriched products
    return enriched_products

# This is used when running with Uvicorn in Docker
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)