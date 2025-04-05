# Use a slim base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

########################### Isabell ################################
# Install system dependencies (including Git for DVC)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*
####################################################################

# Copy app code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

########################### Isabell ################################
RUN pip install --no-cache-dir dvc dagshub
###################################################################

# Download NLTK stopwords
RUN python -m nltk.downloader stopwords

# Expose port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "infer:app", "--host", "0.0.0.0", "--port", "8000"]