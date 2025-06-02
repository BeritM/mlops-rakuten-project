# Use a slim base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including Git for DVC and curl for health check)
RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl \
 && rm -rf /var/lib/apt/lists/*

# copy requirements.txt
COPY dockerfiles/predict_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
  && pip install --no-cache-dir "dvc[s3]"

COPY . .

# Download NLTK stopwords
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('punkt_tab')"

# Expose port
EXPOSE 8002

# Run the FastAPI app
CMD ["uvicorn", "plugins.cd4ml.inference.predict_service:predict_app", "--host", "0.0.0.0", "--port", "8002"]

# Health-Check for Predict Service
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD curl --silent --fail http://localhost:8002/health || exit 1