# Import FastAPI (the web framework) and HTTPException (to return HTTP error responses like 400/500)
from fastapi import FastAPI, HTTPException
# BaseModel = base class for request validation; Field = adds constraints (length, range, defaults)
from pydantic import BaseModel, Field
# Import the functions and sample data from the engine module inside the "backend" package
from backend.engine import SAMPLE_TICKETS, answer_question, classify_ticket, retrieve_knowledge

# Create the FastAPI application object; uvicorn will serve this object
app = FastAPI(
    # Title shown at the top of the auto-generated docs page (/docs)
    title="SupportOps Intelligence API",
    # Short description shown under the title in /docs
    description="Lightweight Hugging Face semantic ticket classification and knowledge retrieval API.",
    # API version shown in /docs
    version="1.0.0",
# End of the FastAPI constructor
)

# Schema for the body of POST /tickets/analyze (FastAPI validates incoming JSON against it)
class TicketRequest(BaseModel):
    # subject is optional (defaults to empty string) and limited to 500 characters
    subject: str = Field(default="", max_length=500)
    # description is required, 3 to 10000 characters; shorter or longer returns HTTP 422 automatically
    description: str = Field(min_length=3, max_length=10000)

# Schema for the body of the two knowledge endpoints
class QuestionRequest(BaseModel):
    # question is required, 3 to 5000 characters
    question: str = Field(min_length=3, max_length=5000)
    # top_k = number of articles to use; default 3, must be between 1 and 5 (ge = >=, le = <=)
    top_k: int = Field(default=3, ge=1, le=5)

# Register this function for HTTP GET requests to "/"
@app.get("/")
# Root endpoint, handy as a quick "is it alive?" check
def root():
    # Return a dict; FastAPI converts it to JSON automatically
    return {"name": "SupportOps Intelligence API", "status": "running", "docs": "/docs"}

# Register this function for GET /health
@app.get("/health")
# Health-check endpoint (used by Docker, load balancers, monitoring)
def health():
    # Report status and which model is in use (hardcoded string, does not actually load the model)
    return {"status": "ok", "model": "sentence-transformers/all-MiniLM-L6-v2"}

# Register this function for GET /tickets/sample
@app.get("/tickets/sample")
# Returns the demo tickets so a UI can show examples
def sample_tickets():
    # Wrap the list in a dict under the key "tickets"
    return {"tickets": SAMPLE_TICKETS}

# Register this function for POST /tickets/analyze
@app.post("/tickets/analyze")
# "payload" is the validated JSON body, parsed into a TicketRequest object
def analyze_ticket(payload: TicketRequest):
    # Start a protected block so errors can be converted to HTTP responses
    try:
        # Run the classifier from engine.py and return its dict as JSON
        return classify_ticket(payload.subject, payload.description)
    # Catch ValueError (the engine raises it for empty/invalid input)
    except ValueError as exc:
        # Return HTTP 400 (bad request) with the error message; "from exc" keeps the original traceback chained
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Catch any other unexpected error (model load failure, memory, etc.)
    except Exception as exc:
        # Return HTTP 500 (server error) with the exception type and message
        raise HTTPException(status_code=500, detail=f"Ticket analysis failed: {type(exc).__name__}: {exc}") from exc

# Register this function for POST /knowledge/search
@app.post("/knowledge/search")
# Receives a validated QuestionRequest body
def search_knowledge(payload: QuestionRequest):
    # Start a protected block
    try:
        # Return the raw top-k matching articles with similarity scores, wrapped under "results"
        return {"results": retrieve_knowledge(payload.question, payload.top_k)}
    # Bad input from the engine -> 400
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Anything else -> 500
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Knowledge search failed: {type(exc).__name__}: {exc}") from exc

# Register this function for POST /knowledge/ask
@app.post("/knowledge/ask")
# Receives a validated QuestionRequest body
def ask_knowledge(payload: QuestionRequest):
    # Start a protected block
    try:
        # Return the composed answer, its sources, method, and grounding note
        return answer_question(payload.question, payload.top_k)
    # Bad input from the engine -> 400
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Anything else -> 500
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Knowledge retrieval failed: {type(exc).__name__}: {exc}") from exc
