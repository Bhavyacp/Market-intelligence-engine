from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag_engine import MarketRAG
from src.vector_store import VectorSearchEngine

app = FastAPI(title="MarketPulse AI API", version="1.0")

rag_engine = MarketRAG()
vector_engine = VectorSearchEngine()

class SearchQuery(BaseModel):
    query: str

@app.get("/")
def home():
    return {"status": "MarketPulse AI Engine Running"}

@app.post("/analyze")
def analyze_market(request: SearchQuery):
    try:
        report = rag_engine.generate_executive_brief(request.query)
        return {
            "status": "success",
            "query": request.query,
            "executive_report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))