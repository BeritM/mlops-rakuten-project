FROM python:3.11-slim

WORKDIR /app

# Systempakete installieren
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

# Pip upgraden + NumPy
RUN pip install --no-cache-dir --upgrade pip numpy==1.23.5

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

CMD ["python", "run_preprocessing.py"]
