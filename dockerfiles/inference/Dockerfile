# Use a slim base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including Git for DVC)
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# copy requirements.txt
COPY dockerfiles/inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY . .
COPY plugins/cd4ml/inference /app/plugins/cd4ml/inference

# Download NLTK stopwords
RUN python -m nltk.downloader stopwords

# Expose port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "plugins.cd4ml.inference.infer:app", "--host", "0.0.0.0", "--port", "8000"]
