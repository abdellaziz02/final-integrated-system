import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import httpx
import asyncio

app = FastAPI(title="Main API Gateway")

# Service URLs from environment variables
SEARCH_ENGINE_URL = os.environ.get("SEARCH_ENGINE_URL")
SCRAPER_URL = os.environ.get("SCRAPER_SERVICE_URL")
PRODUCT_AI_URL = os.environ.get("PRODUCT_AI_SERVICE_URL")
ENRICHER_URL = os.environ.get("ENRICHER_SERVICE_URL")

class UserQueryRequest(BaseModel):
    query: str

async def process_single_url(client: httpx.AsyncClient, url: str) -> List[dict]:
    """
    This function performs the 'deep enrichment' phase on a single,
    already validated supplier URL.
    """
    try:
        print(f"GATEWAY (Phase 2): Starting deep scrape for URL: {url}")
        
        # Call the specialist workers in order
        scraper_res = await client.post(f"{SCRAPER_URL}/scrape/", json={"url": url})
        scraper_res.raise_for_status()
        html_data = scraper_res.json()

        product_ai_res = await client.post(f"{PRODUCT_AI_URL}/identify_products/", json=html_data)
        product_ai_res.raise_for_status()
        raw_products = product_ai_res.json()

        if not raw_products: return []

        enricher_res = await client.post(f"{ENRICHER_URL}/enrich/", json={"products": raw_products})
        enricher_res.raise_for_status()
        return enricher_res.json()
        
    except Exception as e:
        print(f"Warning: Failed to deep scrape/enrich URL {url}: {str(e)}")
        return []

@app.post("/discover-and-enrich")
async def discover_and_enrich_flow(request: UserQueryRequest):
    """
    The main, end-to-end workflow that uses the two-phase discovery process.
    """
    async with httpx.AsyncClient(timeout=600.0) as client: # Increased timeout for the full flow
        try:
            # === PHASE 1: HIGH-LEVEL DISCOVERY ===
            # The gateway calls the search engine, which does the initial NLP call,
            # Google search, and ML-based filtering.
            print(f"GATEWAY (Phase 1): Calling Search Engine to find and validate supplier URLs...")
            
            search_engine_res = await client.post(f"{SEARCH_ENGINE_URL}/api/discover", json={"query": request.query})
            search_engine_res.raise_for_status()
            
            search_data = search_engine_res.json()
            validated_urls = search_data.get("supplier_urls", [])
            print(f"GATEWAY (Phase 1): Search Engine returned {len(validated_urls)} high-potential supplier URLs.")

            if not validated_urls:
                return {
                    "message": "Phase 1 complete. No high-potential supplier URLs were found.",
                    "original_query": request.query,
                    "processed_query_used": search_data.get("processed_query_sent_to_google", ""),
                    "discovered_and_enriched_data": []
                }

            # === PHASE 2: DEEP ENRICHMENT ===
            print(f"GATEWAY (Phase 2): Starting parallel deep enrichment for {len(validated_urls)} URLs...")
            
            processing_tasks = [process_single_url(client, url) for url in validated_urls]
            results_list = await asyncio.gather(*processing_tasks)
            
            final_enriched_data = [product for sublist in results_list for product in sublist]

            print("GATEWAY: Full workflow complete.")
            return {
                "original_query": request.query,
                "processed_query_used": search_data.get("processed_query_sent_to_google", ""),
                "validated_supplier_urls": validated_urls,
                "discovered_and_enriched_data": final_enriched_data
            }
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Error from a downstream service '{e.request.url}': {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred in the gateway: {str(e)}")