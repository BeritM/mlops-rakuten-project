# Use a slim base image
FROM python:3.11-slim

# Set working directory for the build context initially
WORKDIR /app

# Install system dependencies, including Git and gosu
RUN apt-get update \
  && apt-get install -y --no-install-recommends git gosu \
  && rm -rf /var/lib/apt/lists/*

# Git safe directory configuration
RUN git config --global --add safe.directory /app && \
    git config --global --add safe.directory '*'

# Upgrade pip and install dvc[s3]
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir "dvc[s3]"

# Copy requirements.txt and install Python dependencies
COPY dockerfiles/preprocessing/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entrypoint script and make it executable
COPY dockerfiles/preprocessing/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Copy all project files into the container
COPY . .

# Set PYTHONPATH for possibility to use other scripts
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Download NLTK data during build so tokenizers and wordlists are ready at runtime
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

# Set the entrypoint to our script, and the default command
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "plugins/cd4ml/data_processing/run_preprocessing.py"]