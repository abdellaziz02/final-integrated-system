import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    if not client.api_key:
        raise ValueError("GROQ_API_KEY not found.")
except Exception as e:
    print(f"CRITICAL ERROR Initializing Groq Client: {e}")
    client = None

def process_query_with_llm(text: str) -> dict:
    if not client:
        return {}

    system_prompt = """
You are a world-class B2B sourcing expert and multilingual data structuring AI.
Your sole job is to analyze a user's messy query and respond with a single, valid JSON object.
The JSON must have two top-level keys: "english" and "french".
Each of these keys will contain an object with three keys:
1. "product": The clean name of the product.
2. "attributes": A list of extracted attributes.
3. "b2b_search_queries": A list of 3-4 diverse, professional Google search queries an expert would use to find wholesale suppliers of this product.

Your entire response must be ONLY the JSON object.
"""

    user_prompt = f"""
Here is an example of the required output format:
---
User Query: "pomee de terra 5kg bio"
JSON Response: {{
  "english": {{
    "product": "potato",
    "attributes": ["5kg", "organic"],
    "b2b_search_queries": [
      "wholesale 5kg organic potato suppliers",
      "bulk organic potato distributors",
      "B2B potato pricing for food service"
    ]
  }},
  "french": {{
    "product": "pomme de terre",
    "attributes": ["5kg", "bio"],
    "b2b_search_queries": [
      "fournisseur pomme de terre bio 5kg en gros",
      "distributeur pommes de terre vrac",
      "prix B2B pommes de terre pour restauration"
    ]
  }}
}}
---

Now, process the following query. Remember, respond with ONLY the JSON object.

User Query: "{text}"
JSON Response:
"""

    try:
        print(f"DEBUG (NLP Service): Sending prompt to Groq for query: '{text}'")
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama3-70b-8192",
            temperature=0.2,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        
        raw_output = chat_completion.choices[0].message.content
        print(f"DEBUG (NLP Service): Raw LLM output: '{raw_output}'")
        
        return json.loads(raw_output)

    except Exception as e:
        print(f"CRITICAL ERROR processing Groq API call: {e}")
        return {}