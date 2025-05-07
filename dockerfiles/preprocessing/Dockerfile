FROM python:3.11-slim

WORKDIR /app

# Install git (required by DVC)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip


# Copy requirements.txt
COPY dockerfiles/preprocessing/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Set PYTHONPATH for possibility to use other scripts
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Download NLTK data during build so tokenizers and wordlists are ready at runtime – this is best practice because it ensures a deterministic, offline-capable image and faster container startup.
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"


# Set the command to run the preprocessing script when the container starts
CMD ["python", "plugins/cd4ml/data_processing/run_preprocessing.py"]