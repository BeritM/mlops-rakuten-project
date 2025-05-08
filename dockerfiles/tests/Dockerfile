# dockerfiles/tests/Dockerfile

FROM python:3.11-slim
WORKDIR /app

# System‑Deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

# Requirements kopieren & installieren (inkl. vorläufiges numpy 2.x)
COPY dockerfiles/tests/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# NLTK-Stopwords
RUN python -m nltk.downloader punkt stopwords punkt_tab wordnet

# Code rein, shared_volume kopieren
COPY . .
RUN cp -R shared_volume/data /app/ && cp -R shared_volume/models /app/

# Standardkommando
CMD ["pytest", "plugins/cd4ml/tests/test_inference.py", "-v", "-rA"]