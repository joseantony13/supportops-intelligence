"""Lightweight Hugging Face-powered support ticket intelligence engine."""
# Makes type hints lazy-evaluated (lets you write list[str], tuple[...] on older Python versions)
from __future__ import annotations

# lru_cache stores a function's result so it is computed only once (used for caching the model/vectors)
from functools import lru_cache
# Any = "could be any type", used in type hints for flexible dictionaries
from typing import Any
# re = regular expressions, used for cleaning whitespace
import re
# numpy = numerical arrays and math (dot products, argmax, sorting)
import numpy as np
# SentenceTransformer = Hugging Face library that turns text into embedding vectors
from sentence_transformers import SentenceTransformer

# Name of the pretrained model on Hugging Face (small, fast, CPU-friendly, 384-dim embeddings)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Comment: the knowledge base is sample data, replace with real company docs
# Small, fictional knowledge base. Replace or extend with company-approved documents.
# KNOWLEDGE_BASE is a list of dictionaries; each dict is one help article
KNOWLEDGE_BASE = [
    # KB-001: id, title, category, and the article body text for duplicate payments
    {"id": "KB-001", "title": "Duplicate payment troubleshooting", "category": "Billing", "text": "..."},
    # KB-002: password reset / account recovery guidance
    {"id": "KB-002", "title": "Password reset and account recovery", "category": "Account Access", "text": "..."},
    # KB-003: app crash troubleshooting steps
    {"id": "KB-003", "title": "App crash and technical troubleshooting", "category": "Technical Issue", "text": "..."},
    # KB-004: delayed delivery and tracking guidance
    {"id": "KB-004", "title": "Delayed delivery and tracking", "category": "Delivery", "text": "..."},
    # KB-005: refund status and timelines
    {"id": "KB-005", "title": "Refund status and timelines", "category": "Billing", "text": "..."},
    # KB-006: suspicious login / account security
    {"id": "KB-006", "title": "Suspicious login and account security", "category": "Account Access", "text": "..."},
    # KB-007: subscription cancellation
    {"id": "KB-007", "title": "Subscription cancellation", "category": "Billing", "text": "..."},
    # KB-008: service outage and connectivity
    {"id": "KB-008", "title": "Service outage and connectivity", "category": "Technical Issue", "text": "..."},
# End of the knowledge base list
]

# Maps each ticket category to a descriptive sentence; these are embedded and used for zero-shot classification
CATEGORY_DESCRIPTIONS = {
    # Billing: payment-related words the model should associate with this category
    "Billing": "payments, duplicate charges, refunds, invoices, subscription fees, cancellation and billing disputes",
    # Technical Issue: bugs, crashes, errors
    "Technical Issue": "bugs, app crashes, errors, broken features, performance, connectivity and technical troubleshooting",
    # Account Access: login and security words
    "Account Access": "login problems, forgotten passwords, locked accounts, suspicious sign-ins and account security",
    # Delivery: shipping words
    "Delivery": "shipping, packages, tracking, late orders, missing parcels and delivery status",
# End of the category dictionary
}

# Demo tickets (T-1001 to T-1008) for testing/UI; each has id, subject, description, customer
SAMPLE_TICKETS = [
    {"id": "T-1001", "subject": "Charged twice for my order", "description": "...", "customer": "Customer A"},  # duplicate charge
    {"id": "T-1002", "subject": "Cannot log in after password reset", "description": "...", "customer": "Customer B"},  # reset email missing
    {"id": "T-1003", "subject": "Mobile app closes immediately", "description": "...", "customer": "Customer C"},  # app crash
    {"id": "T-1004", "subject": "Package has not moved", "description": "...", "customer": "Customer D"},  # stalled tracking
    {"id": "T-1005", "subject": "I think someone accessed my account", "description": "...", "customer": "Customer E"},  # security
    {"id": "T-1006", "subject": "Refund still missing", "description": "...", "customer": "Customer F"},  # refund delay
    {"id": "T-1007", "subject": "Service keeps timing out", "description": "...", "customer": "Customer G"},  # outage/slow
    {"id": "T-1008", "subject": "Cancel recurring plan", "description": "...", "customer": "Customer H"},  # subscription cancel
# End of the sample tickets list
]

# Cache the return value; maxsize=1 because the function takes no arguments, so only one result exists
@lru_cache(maxsize=1)
# Function that returns the embedding model; "-> SentenceTransformer" is the return type hint
def get_model() -> SentenceTransformer:
    # Docstring: load once, reuse for every request
    """Load the compact model once and reuse it for all requests."""
    # Download (first run) / load the model and force it to run on the CPU
    return SentenceTransformer(MODEL_NAME, device="cpu")

# Cache so category embeddings are computed only once
@lru_cache(maxsize=1)
# Returns a tuple: (list of category names, matrix of their embeddings)
def get_category_vectors() -> tuple[list[str], np.ndarray]:
    # Get the cached model
    model = get_model()
    # List of category names in dictionary order
    names = list(CATEGORY_DESCRIPTIONS)
    # Embed each category description; normalize_embeddings=True makes each vector length 1,
    # so a dot product equals cosine similarity
    vectors = model.encode([CATEGORY_DESCRIPTIONS[n] for n in names], normalize_embeddings=True)
    # Return names plus the vectors as a float32 numpy array (smaller memory)
    return names, np.asarray(vectors, dtype=np.float32)

# Cache so knowledge-base embeddings are computed only once
@lru_cache(maxsize=1)
# Returns a matrix of embeddings, one row per KB article
def get_kb_vectors() -> np.ndarray:
    # Get the cached model
    model = get_model()
    # Build one string per article: "Title. Body text"
    texts = [f"{item['title']}. {item['text']}" for item in KNOWLEDGE_BASE]
    # Embed all articles (normalized) and convert to float32 array
    return np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)

# Helper (leading underscore = "internal use") to tidy text
def _clean(text: str) -> str:
    # Replace any run of whitespace (spaces, tabs, newlines) with one space, then trim the ends
    return re.sub(r"\s+", " ", text).strip()

# Classifies a ticket's category and priority; returns a dictionary
def classify_ticket(subject: str, description: str) -> dict[str, Any]:
    # Combine subject + description into one cleaned string
    content = _clean(f"{subject}. {description}")
    # If nothing meaningful remains (only dots/spaces), reject the input
    if not content.strip(". "):
        # Raise an error that the UI/caller can show to the user
        raise ValueError("Enter a ticket subject or description.")
    # Load category names and their embedding matrix
    names, vectors = get_category_vectors()
    # Embed the ticket text; [0] takes the first (only) vector; convert to float32
    query_vector = np.asarray(get_model().encode([content], normalize_embeddings=True)[0], dtype=np.float32)
    # Matrix-vector product = cosine similarity of the ticket against every category
    scores = vectors @ query_vector
    # Index of the highest score (note: this variable is not actually used later)
    best_index = int(np.argmax(scores))
    # Pair each name with its score and sort from highest to lowest
    ranked = sorted(zip(names, scores.tolist()), key=lambda pair: pair[1], reverse=True)
    # Unpack the top-ranked category and its score
    category, score = ranked[0]

    # Lowercase the text so keyword matching is case-insensitive
    text_lower = content.lower()
    # Keywords that indicate urgent / high-impact issues
    high_signals = ["unauthorized", "hacked", "fraud", "security", "charged twice", "duplicate charge", "cannot access", "can't access", "outage", "urgent", "data loss"]
    # Keywords that indicate moderate issues
    medium_signals = ["not working", "failed", "crash", "missing", "late", "refund", "error", "cannot log in", "can't log in"]
    # If ANY high keyword appears in the text...
    if any(signal in text_lower for signal in high_signals):
        # ...set priority High with an explanation (tuple unpacking assigns both variables)
        priority, rationale = "High", "Potential security, payment, access, or urgent-impact signal detected. Human review is recommended."
    # Otherwise, if ANY medium keyword appears...
    elif any(signal in text_lower for signal in medium_signals):
        # ...set priority Medium
        priority, rationale = "Medium", "A service disruption or unresolved transaction signal was detected."
    # Otherwise (no keywords matched)...
    else:
        # ...set priority Normal
        priority, rationale = "Normal", "No configured high-impact signal was detected. Review business context before acting."

    # Return a result dictionary
    return {
        # Best-matching category
        "category": category,
        # Its cosine similarity, rounded to 4 decimals
        "category_confidence": round(float(score), 4),
        # Full ranked list of all categories with similarity scores
        "category_scores": [{"category": name, "similarity": round(float(value), 4)} for name, value in ranked],
        # Rule-based priority label
        "priority": priority,
        # Why that priority was chosen
        "priority_rationale": rationale,
        # The cleaned text that was analyzed
        "ticket_text": content,
        # Which model produced the embeddings
        "model": MODEL_NAME,
        # Reminder that similarity is not a true probability and a human must review
        "review_note": "Similarity is not a calibrated probability. Category and priority suggestions require human review.",
    }

# Finds the knowledge-base articles most similar to a query; top_k = how many to return (default 3)
def retrieve_knowledge(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    # Clean whitespace in the query
    query = _clean(query)
    # If the query is empty...
    if not query:
        # ...raise an error
        raise ValueError("Enter a question or ticket description.")
    # Load the cached KB embedding matrix
    vectors = get_kb_vectors()
    # Embed the query
    query_vector = np.asarray(get_model().encode([query], normalize_embeddings=True)[0], dtype=np.float32)
    # Cosine similarity between the query and every article
    scores = vectors @ query_vector
    # argsort sorts ascending; [::-1] reverses to descending; the slice keeps the top_k
    # (top_k is clamped between 1 and the number of KB articles)
    indices = np.argsort(scores)[::-1][:max(1, min(top_k, len(KNOWLEDGE_BASE)))]
    # Empty list to collect results
    results = []
    # Loop over the best-matching article indices
    for index in indices:
        # Fetch the article dictionary
        item = KNOWLEDGE_BASE[int(index)]
        # Copy the article's fields and add its similarity score
        results.append({**item, "similarity": round(float(scores[index]), 4)})
    # Return the matches, best first
    return results

# Produces an answer to a question using only retrieved articles
def answer_question(query: str, top_k: int = 3) -> dict[str, Any]:
    # Retrieve the top matching articles
    results = retrieve_knowledge(query, top_k)
    # Comment: no generative LLM is used, so the answer is built from retrieved text
    # Deliberately retrieval-grounded: no generative LLM is used in this low-storage build.
    # Keep only articles with similarity >= 0.20 (a relevance threshold)
    evidence = [item for item in results if item["similarity"] >= 0.20]
    # If nothing passed the threshold...
    if not evidence:
        # ...return a "not found" response with no sources
        return {"answer": "I could not find a sufficiently similar article in the current knowledge base. Please rephrase the question or add an approved troubleshooting document.", "sources": [], "method": "semantic retrieval; no generative LLM", "grounding_note": "No source passed the retrieval threshold. Do not treat this as a resolved answer."}
    # Join the top 2 articles as "Title (ID): text", separated by blank lines
    answer = "\n\n".join([f"{item['title']} ({item['id']}): {item['text']}" for item in evidence[:2]])
    # Return the answer, the source metadata, how it was produced, and a human-verification reminder
    return {"answer": answer, "sources": [{"id": x["id"], "title": x["title"], "category": x["category"], "similarity": x["similarity"]} for x in evidence], "method": "semantic retrieval; no generative LLM", "grounding_note": "The response quotes retrieved knowledge-base guidance; a human should verify it before sending to a customer."}
