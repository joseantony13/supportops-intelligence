from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from backend.engine import SAMPLE_TICKETS, answer_question, classify_ticket, retrieve_knowledge

app = FastAPI(
    title="SupportOps Intelligence API",
    description="Lightweight Hugging Face semantic ticket classification and knowledge retrieval API.",
    version="1.0.0",
)

class TicketRequest(BaseModel):
    subject: str = Field(default="", max_length=500)
    description: str = Field(min_length=3, max_length=10000)

class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=5000)
    top_k: int = Field(default=3, ge=1, le=5)

@app.get("/")
def root():
    return {"name": "SupportOps Intelligence API", "status": "running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "sentence-transformers/all-MiniLM-L6-v2"}

@app.get("/tickets/sample")
def sample_tickets():
    return {"tickets": SAMPLE_TICKETS}

@app.post("/tickets/analyze")
def analyze_ticket(payload: TicketRequest):
    try:
        return classify_ticket(payload.subject, payload.description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ticket analysis failed: {type(exc).__name__}: {exc}") from exc

@app.post("/knowledge/search")
def search_knowledge(payload: QuestionRequest):
    try:
        return {"results": retrieve_knowledge(payload.question, payload.top_k)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Knowledge search failed: {type(exc).__name__}: {exc}") from exc

@app.post("/knowledge/ask")
def ask_knowledge(payload: QuestionRequest):
    try:
        return answer_question(payload.question, payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Knowledge retrieval failed: {type(exc).__name__}: {exc}") from exc
