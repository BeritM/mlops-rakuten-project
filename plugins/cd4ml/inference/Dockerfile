# Use a slim base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy app code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK stopwords
RUN python -m nltk.downloader stopwords

# Expose port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "infer:app", "--host", "0.0.0.0", "--port", "8000"]
