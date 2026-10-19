# Base image: official Python 3.11 on a minimal Debian ("slim" = smaller download, fewer preinstalled tools)
FROM python:3.11-slim

# Set environment variables that apply during the build and inside the running container:
#   PYTHONDONTWRITEBYTECODE=1 -> don't create .pyc cache files (keeps the image smaller)
#   PYTHONUNBUFFERED=1        -> print logs immediately instead of buffering them (so `docker logs` is live)
#   HF_HOME=...               -> tell Hugging Face to store downloaded models under /app/.cache/huggingface
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.cache/huggingface

# Set the working directory; all later commands run from /app (created automatically if missing)
WORKDIR /app

# Copy ONLY requirements.txt first, so Docker can cache the install step below
# (if only your code changes, this layer and the pip install are reused)
COPY requirements.txt .

# Install Python dependencies; --no-cache-dir skips pip's download cache to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend package (engine.py, main.py, __init__.py) into /app/backend
COPY backend ./backend

# Copy the Streamlit frontend into /app/frontend
COPY frontend ./frontend

# Document which ports the container listens on: 8000 = FastAPI, 8501 = Streamlit
# (this is documentation only; it does NOT publish the ports, that's done with `-p` at run time)
EXPOSE 8000 8501

# Default command when the container starts: run the FastAPI backend with uvicorn
#   0.0.0.0 = listen on all interfaces, so it's reachable from outside the container
# Written in "exec form" (a JSON list) so uvicorn receives stop signals correctly
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
