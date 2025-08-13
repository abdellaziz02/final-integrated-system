from pydantic import BaseModel, Field
from typing import List, Optional

class LanguageOutput(BaseModel):
    """A nested model to hold all data for a single language."""
    product: str
    attributes: List[str]
    b2b_search_queries: List[str]

class SearchData(BaseModel):
    """The main data structure containing both English and French outputs."""
    english: LanguageOutput
    french: LanguageOutput

class QueryRequest(BaseModel):
    query: str

class FinalResponse(BaseModel):
    original_query: str
    search_data: Optional[SearchData] = None