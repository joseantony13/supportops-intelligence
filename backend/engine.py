"""Lightweight Hugging Face-powered support ticket intelligence engine."""
from __future__ import annotations

from functools import lru_cache
from typing import Any
import re
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Small, fictional knowledge base. Replace or extend with company-approved documents.
KNOWLEDGE_BASE = [
    {"id": "KB-001", "title": "Duplicate payment troubleshooting", "category": "Billing", "text": "If a customer reports being charged twice, first check whether one transaction is pending authorization. Compare transaction IDs and timestamps. Pending authorizations often disappear automatically within 3 to 7 business days depending on the bank. If both transactions are posted, escalate to the billing team with receipts and transaction references. Never ask a customer to share a full card number or security code."},
    {"id": "KB-002", "title": "Password reset and account recovery", "category": "Account Access", "text": "For a forgotten password, direct the customer to the official password reset page and confirm they can access their registered email. Ask them to check spam or junk folders. Do not ask for passwords or one-time passcodes. If the email address is no longer accessible or suspicious sign-in activity is reported, route the case to the account security team."},
    {"id": "KB-003", "title": "App crash and technical troubleshooting", "category": "Technical Issue", "text": "For an app crash, collect the device type, operating system, app version, approximate time, and steps that reproduce the problem. Ask the customer to restart the app, check for an update, and retry on a stable connection. Do not ask the customer to remove data before checking whether it is backed up. Escalate reproducible crashes with logs that do not contain secrets or personal data."},
    {"id": "KB-004", "title": "Delayed delivery and tracking", "category": "Delivery", "text": "For a delayed shipment, verify the order reference and review the latest carrier scan. Explain that tracking can remain unchanged between scans. If the shipment is beyond the stated delivery window, open a carrier investigation and provide the case reference. Do not promise an exact arrival date unless the carrier confirms it."},
    {"id": "KB-005", "title": "Refund status and timelines", "category": "Billing", "text": "When a refund is approved, share the refund reference and the expected processing window from the payment provider. Banks may take several business days to display the credit. Confirm the original payment method and refund status in the internal system. Escalate if the stated processing window has passed. Do not claim a refund was completed unless the transaction status confirms it."},
    {"id": "KB-006", "title": "Suspicious login and account security", "category": "Account Access", "text": "Treat reports of unauthorized access as high priority. Advise the customer to use the official account recovery flow, change their password from a trusted device, and enable multi-factor authentication if available. Never request a password or one-time code. Escalate suspected account compromise to security and preserve relevant timestamps and non-sensitive evidence."},
    {"id": "KB-007", "title": "Subscription cancellation", "category": "Billing", "text": "To cancel a subscription, guide the customer to the official subscription settings and explain when access ends. Check whether the subscription was purchased through a third-party app store, because cancellation may need to happen there. Cancellation does not automatically imply a refund; review the applicable refund policy separately."},
    {"id": "KB-008", "title": "Service outage and connectivity", "category": "Technical Issue", "text": "When multiple users report the same service failure, check the official service status page before troubleshooting each device individually. Record the affected feature, start time, region if relevant, and error message. Communicate only confirmed incident details and avoid promising a resolution time that has not been published."},
]

CATEGORY_DESCRIPTIONS = {
    "Billing": "payments, duplicate charges, refunds, invoices, subscription fees, cancellation and billing disputes",
    "Technical Issue": "bugs, app crashes, errors, broken features, performance, connectivity and technical troubleshooting",
    "Account Access": "login problems, forgotten passwords, locked accounts, suspicious sign-ins and account security",
    "Delivery": "shipping, packages, tracking, late orders, missing parcels and delivery status",
}

SAMPLE_TICKETS = [
    {"id": "T-1001", "subject": "Charged twice for my order", "description": "My bank statement shows two charges for the same order. Please check this urgently.", "customer": "Customer A"},
    {"id": "T-1002", "subject": "Cannot log in after password reset", "description": "The reset email never arrives and I cannot access my account.", "customer": "Customer B"},
    {"id": "T-1003", "subject": "Mobile app closes immediately", "description": "The app crashes every time I open the dashboard after the latest update.", "customer": "Customer C"},
    {"id": "T-1004", "subject": "Package has not moved", "description": "Tracking has not updated for four days and the delivery date has passed.", "customer": "Customer D"},
    {"id": "T-1005", "subject": "I think someone accessed my account", "description": "I received an unfamiliar sign-in alert and did not make that login.", "customer": "Customer E"},
    {"id": "T-1006", "subject": "Refund still missing", "description": "My refund was approved last week but I cannot see the credit in my account.", "customer": "Customer F"},
    {"id": "T-1007", "subject": "Service keeps timing out", "description": "Several pages are showing errors and the website is very slow today.", "customer": "Customer G"},
    {"id": "T-1008", "subject": "Cancel recurring plan", "description": "I want to stop my monthly subscription and understand when it ends.", "customer": "Customer H"},
]

@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the compact model once and reuse it for all requests."""
    return SentenceTransformer(MODEL_NAME, device="cpu")

@lru_cache(maxsize=1)
def get_category_vectors() -> tuple[list[str], np.ndarray]:
    model = get_model()
    names = list(CATEGORY_DESCRIPTIONS)
    vectors = model.encode([CATEGORY_DESCRIPTIONS[n] for n in names], normalize_embeddings=True)
    return names, np.asarray(vectors, dtype=np.float32)

@lru_cache(maxsize=1)
def get_kb_vectors() -> np.ndarray:
    model = get_model()
    texts = [f"{item['title']}. {item['text']}" for item in KNOWLEDGE_BASE]
    return np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def classify_ticket(subject: str, description: str) -> dict[str, Any]:
    content = _clean(f"{subject}. {description}")
    if not content.strip(". "):
        raise ValueError("Enter a ticket subject or description.")
    names, vectors = get_category_vectors()
    query_vector = np.asarray(get_model().encode([content], normalize_embeddings=True)[0], dtype=np.float32)
    scores = vectors @ query_vector
    best_index = int(np.argmax(scores))
    ranked = sorted(zip(names, scores.tolist()), key=lambda pair: pair[1], reverse=True)
    category, score = ranked[0]

    text_lower = content.lower()
    high_signals = ["unauthorized", "hacked", "fraud", "security", "charged twice", "duplicate charge", "cannot access", "can't access", "outage", "urgent", "data loss"]
    medium_signals = ["not working", "failed", "crash", "missing", "late", "refund", "error", "cannot log in", "can't log in"]
    if any(signal in text_lower for signal in high_signals):
        priority, rationale = "High", "Potential security, payment, access, or urgent-impact signal detected. Human review is recommended."
    elif any(signal in text_lower for signal in medium_signals):
        priority, rationale = "Medium", "A service disruption or unresolved transaction signal was detected."
    else:
        priority, rationale = "Normal", "No configured high-impact signal was detected. Review business context before acting."

    return {
        "category": category,
        "category_confidence": round(float(score), 4),
        "category_scores": [{"category": name, "similarity": round(float(value), 4)} for name, value in ranked],
        "priority": priority,
        "priority_rationale": rationale,
        "ticket_text": content,
        "model": MODEL_NAME,
        "review_note": "Similarity is not a calibrated probability. Category and priority suggestions require human review.",
    }

def retrieve_knowledge(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    query = _clean(query)
    if not query:
        raise ValueError("Enter a question or ticket description.")
    vectors = get_kb_vectors()
    query_vector = np.asarray(get_model().encode([query], normalize_embeddings=True)[0], dtype=np.float32)
    scores = vectors @ query_vector
    indices = np.argsort(scores)[::-1][:max(1, min(top_k, len(KNOWLEDGE_BASE)))]
    results = []
    for index in indices:
        item = KNOWLEDGE_BASE[int(index)]
        results.append({**item, "similarity": round(float(scores[index]), 4)})
    return results

def answer_question(query: str, top_k: int = 3) -> dict[str, Any]:
    results = retrieve_knowledge(query, top_k)
    # Deliberately retrieval-grounded: no generative LLM is used in this low-storage build.
    evidence = [item for item in results if item["similarity"] >= 0.20]
    if not evidence:
        return {"answer": "I could not find a sufficiently similar article in the current knowledge base. Please rephrase the question or add an approved troubleshooting document.", "sources": [], "method": "semantic retrieval; no generative LLM", "grounding_note": "No source passed the retrieval threshold. Do not treat this as a resolved answer."}
    answer = "\n\n".join([f"{item['title']} ({item['id']}): {item['text']}" for item in evidence[:2]])
    return {"answer": answer, "sources": [{"id": x["id"], "title": x["title"], "category": x["category"], "similarity": x["similarity"]} for x in evidence], "method": "semantic retrieval; no generative LLM", "grounding_note": "The response quotes retrieved knowledge-base guidance; a human should verify it before sending to a customer."}
