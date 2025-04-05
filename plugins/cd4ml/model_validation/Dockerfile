FROM python:3.11-slim

WORKDIR /app

COPY model_validation.py /app/
COPY model_training.py /app/
COPY run_model_validation.py /app/
COPY requirements-model.txt /app/

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements-model.txt && \
    pip install --no-cache-dir \
    dvc \
    dvc[s3] \
    dagshub

ENV MODEL_OUTPUT_DIR=/app/models
ENV DATA_INPUT_DIR=/app/data/processed

CMD ["python", "run_model_validation.py"]