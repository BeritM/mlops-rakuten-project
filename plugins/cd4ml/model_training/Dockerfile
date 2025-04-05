FROM python:3.11-slim

WORKDIR /app

COPY requirements-model.txt /app/
COPY run_model_training.py /app/
COPY model_training.py /app/


RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-model.txt && \
    pip install --no-cache-dir \
    dvc \
    dagshub \
    pandas \
    mlflow \
    joblib \
    scikit-learn


ENV MODEL_OUTPUT_DIR=/app/models
ENV DATA_INPUT_DIR=/app/data/processed

CMD ["python", "run_model_training.py"]