FROM python:3.10-slim

WORKDIR /app

# Install git (required by DVC)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install DVC and additional Python dependencies directly
RUN pip install --no-cache-dir \
    dvc \
    pandas \
    scikit-learn \
    dagshub \
    mlflow \
    joblib

# Copy requirements.txt and install additional Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Import nltk and download required datasets
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

# Set the command to run the preprocessing script when the container starts
CMD ["python", "run_preprocessing.py"]
