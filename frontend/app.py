# os = read environment variables
import os
# requests = make HTTP calls to the FastAPI backend
import requests
# pandas = tables (DataFrames) for display, filtering and CSV export
import pandas as pd
# streamlit = the web UI framework; "st" is the conventional alias
import streamlit as st

# Backend URL: use the SUPPORTOPS_API_URL environment variable if set, otherwise localhost port 8000
API_URL = os.getenv("SUPPORTOPS_API_URL", "http://127.0.0.1:8000")

# Configure the browser tab title, tab icon and wide page layout (must be the first Streamlit call)
st.set_page_config(page_title="SupportOps Intelligence", page_icon="🧠", layout="wide")
# Big page heading
st.title("🧠 SupportOps Intelligence")
# Small grey subtitle under the heading
st.caption("Hugging Face semantic ticket intelligence · knowledge retrieval · human-in-the-loop workflow")
# Blue info box reminding viewers that the data is fictional and the scores are not true probabilities
st.info("This portfolio demo uses fictional sample tickets. Category confidence is similarity, not a calibrated probability. Priority rules are heuristics, not a trained risk model.")

# Cache this function's result for 30 seconds so the API isn't called on every rerun
@st.cache_data(ttl=30)
# Fetches the demo tickets from the backend
def get_samples():
    # GET /tickets/sample with a 10-second timeout
    response = requests.get(f"{API_URL}/tickets/sample", timeout=10)
    # Raise an exception if the status code is 4xx/5xx
    response.raise_for_status()
    # Parse the JSON and return just the list under the "tickets" key
    return response.json()["tickets"]

# Reusable helper for POST requests to the API
def api_post(path, payload):
    # Send payload as JSON; 180-second timeout because the first call may load/download the model
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=180)
    # Raise an exception on error status codes
    response.raise_for_status()
    # Return the parsed JSON response
    return response.json()

# Everything indented under this block is drawn in the left sidebar
with st.sidebar:
    # Sidebar heading
    st.header("Navigation")
    # Radio buttons to pick a page; the selected label is stored in "page"
    page = st.radio("Choose a workspace", ["Overview", "Analyze a ticket", "Knowledge assistant", "Ticket explorer"])
    # Horizontal divider line
    st.divider()
    # Small label
    st.caption("Backend API")
    # Show the API URL in a code box
    st.code(API_URL)
    # Button that stretches to the sidebar width; True when clicked
    if st.button("Check API health", use_container_width=True):
        # Protected block for network errors
        try:
            # Call GET /health with a 5-second timeout and parse the JSON
            health = requests.get(f"{API_URL}/health", timeout=5).json()
            # Green success box showing the status
            st.success(f"API status: {health['status']}")
            # Show the model name in small text
            st.caption(health["model"])
        # Catch connection errors, timeouts, etc.
        except requests.RequestException as exc:
            # Red error box
            st.error(f"API is not reachable: {exc}")

# PAGE 1: Overview
# Runs only when "Overview" is selected
if page == "Overview":
    # Protected block in case the backend is down
    try:
        # Fetch the sample tickets and convert them to a DataFrame
        tickets = pd.DataFrame(get_samples())
        # Create three side-by-side columns
        col1, col2, col3 = st.columns(3)
        # Metric card: number of demo tickets
        col1.metric("Demo tickets", len(tickets))
        # Metric card: knowledge articles (hardcoded to 8, matching the backend)
        col2.metric("Knowledge articles", 8)
        # Metric card: embedding size of the MiniLM model (hardcoded)
        col3.metric("Semantic embedding size", "384 dimensions")
        # Section heading
        st.subheader("Sample ticket queue")
        # Table of selected columns; full width, row index hidden
        st.dataframe(tickets[["id", "subject", "description"]], use_container_width=True, hide_index=True)
        # Section heading
        st.subheader("System architecture")
        # One-line architecture flow rendered as markdown (backticks make code-style chips)
        st.markdown("`Ticket / Question` → `FastAPI` → `Hugging Face MiniLM embeddings` → `Cosine similarity` → `Classification / Retrieved sources` → `Streamlit UI`")
        # Section heading
        st.subheader("What makes this portfolio-worthy")
        # Bullet list ("\n" separates the bullets)
        st.markdown("- Reusable API endpoints with request validation\n- Semantic similarity instead of keyword-only matching\n- Knowledge retrieval with visible source IDs\n- Dashboard, error handling, and human-review notes\n- Docker-ready deployment structure")
    # If the API call failed...
    except requests.RequestException as exc:
        # ...tell the user to start the backend
        st.error(f"Cannot connect to the backend. Start the API first. Details: {exc}")

#  PAGE 2: Analyze a ticket 
# Runs only when "Analyze a ticket" is selected
elif page == "Analyze a ticket":
    # Section heading
    st.subheader("Ticket triage")
    # A form groups inputs so the app reruns only when the submit button is pressed
    with st.form("ticket_form"):
        # Single-line input for the subject
        subject = st.text_input("Ticket subject", placeholder="Charged twice for my order")
        # Multi-line input for the description, 150 px tall
        description = st.text_area("Ticket description", height=150, placeholder="Describe the customer issue...")
        # Submit button (primary = highlighted); True when clicked
        submitted = st.form_submit_button("Analyze ticket", type="primary")
    # Only run when the form was submitted
    if submitted:
        # Client-side check that mirrors the backend's min_length=3 rule
        if len(description.strip()) < 3:
            # Yellow warning box
            st.warning("Enter a description with at least 3 characters.")
        # Otherwise call the API
        else:
            # Protected block for network/HTTP errors
            try:
                # POST to /tickets/analyze and store the result dict
                result = api_post("/tickets/analyze", {"subject": subject, "description": description})
                # Three columns for the headline metrics
                c1, c2, c3 = st.columns(3)
                # Predicted category
                c1.metric("Suggested category", result["category"])
                # Rule-based priority
                c2.metric("Priority suggestion", result["priority"])
                # Top cosine similarity formatted to 3 decimals
                c3.metric("Top similarity", f"{result['category_confidence']:.3f}")
                # Bold label (markdown **)
                st.write("**Why this priority was suggested**")
                # Show the explanation text
                st.write(result["priority_rationale"])
                # Bold label
                st.write("**Category similarity ranking**")
                # Convert the list of {category, similarity} dicts to a DataFrame
                scores = pd.DataFrame(result["category_scores"])
                # Bar chart with category on the x-axis and similarity on the y-axis
                st.bar_chart(scores.set_index("category")["similarity"])
                # Small-text reminder that a human must review
                st.caption(result["review_note"])
                # Show the raw JSON result, excluding the already-charted category_scores
                st.json({k: v for k, v in result.items() if k not in ["category_scores"]})
            # If the API call failed...
            except requests.RequestException as exc:
                # ...show the error
                st.error(f"API request failed: {exc}")

# PAGE 3: Knowledge assistant
# Runs only when "Knowledge assistant" is selected
elif page == "Knowledge assistant":
    # Section heading
    st.subheader("Ask the support knowledge base")
    # Explain that this mode retrieves articles and does not generate text
    st.write("This lightweight mode retrieves relevant approved articles and shows their source IDs. It does not use a large generative LLM.")
    # Multi-line question box, 100 px tall
    question = st.text_area("Question", placeholder="A customer was charged twice. What should support check?", height=100)
    # Slider for how many articles to retrieve (1 to 5, default 3)
    top_k = st.slider("Maximum articles to retrieve", min_value=1, max_value=5, value=3)
    # Button; True when clicked
    if st.button("Find evidence", type="primary"):
        # Client-side validation
        if len(question.strip()) < 3:
            # Warn the user
            st.warning("Enter a longer question.")
        # Otherwise call the API
        else:
            # Protected block
            try:
                # POST to /knowledge/ask
                result = api_post("/knowledge/ask", {"question": question, "top_k": top_k})
                # Heading for the answer
                st.subheader("Retrieved guidance")
                # Show the combined article text
                st.write(result["answer"])
                # Small-text note saying a human should verify it
                st.caption(result["grounding_note"])
                # Heading for the sources table
                st.subheader("Sources")
                # If the list of sources is not empty...
                if result["sources"]:
                    # ...show it as a table
                    st.dataframe(pd.DataFrame(result["sources"]), use_container_width=True, hide_index=True)
                # Otherwise (empty list)...
                else:
                    # ...warn that nothing matched
                    st.warning("No sufficiently similar article was found.")
            # If the API call failed...
            except requests.RequestException as exc:
                # ...show the error
                st.error(f"API request failed: {exc}")

# PAGE 4: Ticket explorer
# Runs only when "Ticket explorer" is selected
elif page == "Ticket explorer":
    # Section heading
    st.subheader("Explore demo tickets")
    # Protected block
    try:
        # Load sample tickets into a DataFrame
        df = pd.DataFrame(get_samples())
        # Text box for filtering
        search = st.text_input("Filter tickets", placeholder="Search subject or description")
        # Only filter if the user typed something
        if search:
            # Convert every cell to string, test if it contains the search text
            # (case-insensitive, plain text not regex), and keep rows where ANY column matches
            mask = df.astype(str).apply(lambda col: col.str.contains(search, case=False, regex=False)).any(axis=1)
            # Keep only the matching rows
            df = df[mask]
        # Show the (possibly filtered) table
        st.dataframe(df, use_container_width=True, hide_index=True)
        # Download button: converts the table to CSV bytes and offers it as support_tickets.csv
        st.download_button("Download displayed tickets as CSV", df.to_csv(index=False).encode("utf-8"), "support_tickets.csv", "text/csv")
    # If loading failed...
    except requests.RequestException as exc:
        # ...show the error
        st.error(f"Cannot load tickets: {exc}")
