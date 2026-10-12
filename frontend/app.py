import os
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("SUPPORTOPS_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="SupportOps Intelligence", page_icon="🧠", layout="wide")
st.title("🧠 SupportOps Intelligence")
st.caption("Hugging Face semantic ticket intelligence · knowledge retrieval · human-in-the-loop workflow")
st.info("This portfolio demo uses fictional sample tickets. Category confidence is similarity, not a calibrated probability. Priority rules are heuristics, not a trained risk model.")

@st.cache_data(ttl=30)
def get_samples():
    response = requests.get(f"{API_URL}/tickets/sample", timeout=10)
    response.raise_for_status()
    return response.json()["tickets"]

def api_post(path, payload):
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=180)
    response.raise_for_status()
    return response.json()

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose a workspace", ["Overview", "Analyze a ticket", "Knowledge assistant", "Ticket explorer"])
    st.divider()
    st.caption("Backend API")
    st.code(API_URL)
    if st.button("Check API health", use_container_width=True):
        try:
            health = requests.get(f"{API_URL}/health", timeout=5).json()
            st.success(f"API status: {health['status']}")
            st.caption(health["model"])
        except requests.RequestException as exc:
            st.error(f"API is not reachable: {exc}")

if page == "Overview":
    try:
        tickets = pd.DataFrame(get_samples())
        col1, col2, col3 = st.columns(3)
        col1.metric("Demo tickets", len(tickets))
        col2.metric("Knowledge articles", 8)
        col3.metric("Semantic embedding size", "384 dimensions")
        st.subheader("Sample ticket queue")
        st.dataframe(tickets[["id", "subject", "description"]], use_container_width=True, hide_index=True)
        st.subheader("System architecture")
        st.markdown("`Ticket / Question` → `FastAPI` → `Hugging Face MiniLM embeddings` → `Cosine similarity` → `Classification / Retrieved sources` → `Streamlit UI`")
        st.subheader("What makes this portfolio-worthy")
        st.markdown("- Reusable API endpoints with request validation\n- Semantic similarity instead of keyword-only matching\n- Knowledge retrieval with visible source IDs\n- Dashboard, error handling, and human-review notes\n- Docker-ready deployment structure")
    except requests.RequestException as exc:
        st.error(f"Cannot connect to the backend. Start the API first. Details: {exc}")

elif page == "Analyze a ticket":
    st.subheader("Ticket triage")
    with st.form("ticket_form"):
        subject = st.text_input("Ticket subject", placeholder="Charged twice for my order")
        description = st.text_area("Ticket description", height=150, placeholder="Describe the customer issue...")
        submitted = st.form_submit_button("Analyze ticket", type="primary")
    if submitted:
        if len(description.strip()) < 3:
            st.warning("Enter a description with at least 3 characters.")
        else:
            try:
                result = api_post("/tickets/analyze", {"subject": subject, "description": description})
                c1, c2, c3 = st.columns(3)
                c1.metric("Suggested category", result["category"])
                c2.metric("Priority suggestion", result["priority"])
                c3.metric("Top similarity", f"{result['category_confidence']:.3f}")
                st.write("**Why this priority was suggested**")
                st.write(result["priority_rationale"])
                st.write("**Category similarity ranking**")
                scores = pd.DataFrame(result["category_scores"])
                st.bar_chart(scores.set_index("category")["similarity"])
                st.caption(result["review_note"])
                st.json({k: v for k, v in result.items() if k not in ["category_scores"]})
            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")

elif page == "Knowledge assistant":
    st.subheader("Ask the support knowledge base")
    st.write("This lightweight mode retrieves relevant approved articles and shows their source IDs. It does not use a large generative LLM.")
    question = st.text_area("Question", placeholder="A customer was charged twice. What should support check?", height=100)
    top_k = st.slider("Maximum articles to retrieve", min_value=1, max_value=5, value=3)
    if st.button("Find evidence", type="primary"):
        if len(question.strip()) < 3:
            st.warning("Enter a longer question.")
        else:
            try:
                result = api_post("/knowledge/ask", {"question": question, "top_k": top_k})
                st.subheader("Retrieved guidance")
                st.write(result["answer"])
                st.caption(result["grounding_note"])
                st.subheader("Sources")
                if result["sources"]:
                    st.dataframe(pd.DataFrame(result["sources"]), use_container_width=True, hide_index=True)
                else:
                    st.warning("No sufficiently similar article was found.")
            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")

elif page == "Ticket explorer":
    st.subheader("Explore demo tickets")
    try:
        df = pd.DataFrame(get_samples())
        search = st.text_input("Filter tickets", placeholder="Search subject or description")
        if search:
            mask = df.astype(str).apply(lambda col: col.str.contains(search, case=False, regex=False)).any(axis=1)
            df = df[mask]
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download displayed tickets as CSV", df.to_csv(index=False).encode("utf-8"), "support_tickets.csv", "text/csv")
    except requests.RequestException as exc:
        st.error(f"Cannot load tickets: {exc}")
