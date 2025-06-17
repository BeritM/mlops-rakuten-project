# Use a slim base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including Git for DVC and curl for health check)
RUN apt-get update \
  && apt-get install -y --no-install-recommends git curl gosu \
  && rm -rf /var/lib/apt/lists/*

# Git safe directory configuration
RUN git config --global --add safe.directory /app && \
    git config --global --add safe.directory '*'

# copy requirements.txt
COPY dockerfiles/predict_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
  && pip install --no-cache-dir "dvc[s3]"

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/share/nltk_data'); nltk.download('punkt', download_dir='/usr/local/share/nltk_data'); nltk.download('wordnet', download_dir='/usr/local/share/nltk_data'); nltk.download('punkt_tab', download_dir='/usr/local/share/nltk_data')"
ENV NLTK_DATA=/usr/local/share/nltk_data

COPY . .

COPY dockerfiles/predict_service/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Expose port
EXPOSE 8002

# Set the entrypoint script - user permissions
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# Run the FastAPI app
CMD ["uvicorn", "plugins.cd4ml.inference.predict_service:predict_app", "--host", "0.0.0.0", "--port", "8002"]

# Health-Check for Predict Service
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD curl --silent --fail http://localhost:8002/health || exit 1