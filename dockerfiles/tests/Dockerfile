FROM python:3.11-slim
WORKDIR /app

# System‑Deps
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Copy & install requirements
COPY dockerfiles/tests/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "dvc[s3]"      # CHANGED

# NLTK-Stopwords
RUN python -m nltk.downloader stopwords

COPY . .
RUN cp -R shared_volume/data /app/ && cp -R shared_volume/models /app/

# Command
CMD ["pytest", "plugins/cd4ml/tests/test_inference.py", "-v", "-rA"]
