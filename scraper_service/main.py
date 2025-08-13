# /scraper_service/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium_stealth import stealth
import httpx # <-- Import httpx

# Initialize the FastAPI app
app = FastAPI(
    title="Optimized Scraping Service",
    description="A microservice to fetch HTML. It tries a fast HTTP request first, then falls back to Selenium."
)

# Pydantic model for incoming request body
class URLPayload(BaseModel):
    url: HttpUrl

@app.post("/scrape/")
async def scrape_url(payload: URLPayload):
    """
    Tries a fast, direct HTTP request first. If it fails or seems
    to be blocked, it uses the slower but more powerful Selenium/Chrome browser.
    """
    url = str(payload.url)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # --- STRATEGY 1: FAST HTTPX REQUEST ---
    try:
        print(f"Attempting fast scrape for: {url}")
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, follow_redirects=True, timeout=15)
            response.raise_for_status() # Will raise an exception for 4xx/5xx responses
            html_content = response.text
            # Simple check: if HTML is very short, it might be a block page.
            if len(html_content) > 500: # You can adjust this threshold
                print(f"✅ Fast scrape successful for {url}")
                return {"html": html_content, "method": "fast"}
            else:
                print("⚠️ Fast scrape returned minimal content, falling back to Selenium.")
    except Exception as e:
        print(f"⚠️ Fast scrape failed ({e}), falling back to Selenium.")

    # --- STRATEGY 2: SLOW SELENIUM FALLBACK ---
    print(f"Executing Selenium fallback for: {url}")
    options = Options()
    options.add_argument("--headless")
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = None # Initialize driver to None
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        stealth(driver, languages=["en-US", "en"], vendor="Google Inc.", platform="Win32", webgl_vendor="Intel Inc.", renderer="Intel Iris OpenGL Engine", fix_hairline=True)
        driver.get(url)
        html_content = driver.page_source
        print(f"✅ Selenium scrape successful for {url}")
        return {"html": html_content, "method": "selenium"}

    except Exception as e:
        print(f"❌ Both fast and Selenium scrapes failed. Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to scrape the URL with all methods. Error: {str(e)}")
    finally:
        if driver:
            driver.quit()