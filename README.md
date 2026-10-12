# SupportOps Intelligence

A lightweight Hugging Face portfolio project for semantic support-ticket categorization, heuristic priority suggestions, and source-grounded knowledge retrieval.

## Stack
- Hugging Face `sentence-transformers/all-MiniLM-L6-v2` embeddings
- NumPy cosine similarity
- FastAPI REST API and interactive OpenAPI docs
- Streamlit dashboard
- Docker-ready backend

## Important scope note
This low-storage build does **not** use a generative LLM. The knowledge assistant retrieves and displays matching knowledge-base passages with source IDs. Ticket categories are selected by similarity to category descriptions; priority is based on configurable rules. Similarity is not calibrated confidence, and outputs require human review.

## Run locally
Use Python 3.11 or 3.12. The first model load downloads model files from Hugging Face; subsequent runs use the local cache.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

In a second VS Code terminal:

```powershell
.venv\Scripts\Activate.ps1
streamlit run frontend/app.py
```

- Dashboard: http://localhost:8501
- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## API examples

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/tickets/analyze -ContentType 'application/json' -Body '{"subject":"Charged twice","description":"My bank shows two charges for the same order."}'
```

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/knowledge/ask -ContentType 'application/json' -Body '{"question":"What should I check if a customer was charged twice?","top_k":3}'
```

## Tests
```powershell
pip install pytest httpx
pytest -q
```

## Extend it
1. Replace fictional articles with company-approved public/sample knowledge documents.
2. Add a labeled ticket dataset and evaluate category accuracy, macro-F1, and confusion matrix.
3. Add persistent vector indexing if the knowledge base grows.
4. Add authentication, rate limits, PII redaction, structured logging, and monitoring before real deployment.
5. Optional: connect a small instruction model or hosted inference API for generated answers, while retaining citations and evaluation.

## License
Code is provided as a portfolio starter. Check the Hugging Face model card for the model's license and usage details.
