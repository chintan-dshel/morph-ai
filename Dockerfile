FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application modules
COPY app.py optimizer.py visualization.py geometry.py \
     materials.py chat.py meshing.py utils.py ./

# Copy Streamlit theme config only — secrets.toml is excluded by .dockerignore.
# Pass API keys at runtime via environment variables:
#   ANTHROPIC_API_KEY=sk-ant-... docker compose up
COPY .streamlit/config.toml .streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
