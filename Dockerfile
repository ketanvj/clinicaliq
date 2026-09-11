FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by sentence-transformers / chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Pre-download the HuggingFace embedding model so the container doesn't
# need internet access at runtime (model is cached in the image layer)
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')"

# Seed the SQLite database and build the ChromaDB vector store
RUN python data/seed.py && python data/ingest.py

EXPOSE 8501

# Streamlit config: disable the browser auto-open and CORS for container use
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "s13/starter/app.py"]
