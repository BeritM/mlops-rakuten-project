FROM python:3.11-slim

# Set working dir
WORKDIR /app

# Install system deps (git for DVC, if needed)
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Copy & install Python dependencies 
COPY dockerfiles/model_training/requirements-model.txt ./requirements-model.txt
RUN pip install --no-cache-dir -r requirements-model.txt

# Copy model_training plugin code
COPY . .
#COPY plugins/cd4ml/model_training /app/plugins/cd4ml/model_training

ENV MODEL_OUTPUT_DIR=/app/models
ENV DATA_INPUT_DIR=/app/data/processed

CMD ["python", "plugins/cd4ml/model_training/run_model_training.py"]