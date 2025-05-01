# dockerfiles/tests/Dockerfile

FROM python:3.11-slim
WORKDIR /app

# System‑Deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

# Requirements kopieren & installieren (inkl. vorläufiges numpy 2.x)
COPY dockerfiles/tests/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir dvc[s3] #### Max ####

# NLTK-Stopwords
RUN python -m nltk.downloader stopwords

# Nur noch den Test-Code in den Container kopieren
# COPY . .  # plugins/cd4ml/tests /app/plugins/cd4ml/tests

# Standardkommando
CMD ["pytest", "plugins/cd4ml/tests/test_inference.py", "-v", "-rA"]