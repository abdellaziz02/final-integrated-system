from fastapi import FastAPI
from app.model.query_model import QueryRequest, FinalResponse, SearchData
from app.service.nlp_pipeline import process_query_with_llm

app = FastAPI(
    title="Agro-Food NLP Microservice",
    description="Processes multilingual queries into structured, expanded B2B search data.",
    version="16.0.0" # THE FINAL QUERY EXPANSION BUILD
)

@app.get("/")
async def root():
    return {"message": "NLP Service is running!"}

@app.post("/process-query", response_model=FinalResponse)
async def process_query_endpoint(request: QueryRequest):
    """
    Receives a raw user query and uses an LLM to generate structured
    search data, including expanded B2B queries in English and French.
    """
    result_dict = process_query_with_llm(request.query)
    
    # Use Pydantic to safely parse the dictionary from the LLM
    search_data = SearchData.model_validate(result_dict) if result_dict else None

    return FinalResponse(
        original_query=request.query,
        search_data=search_data
    )