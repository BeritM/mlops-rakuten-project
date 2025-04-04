FROM python:3.11-slim

WORKDIR /app

COPY plugins/cd4ml/model_training/model_training.py /app/
COPY plugins/cd4ml/model_training/run_model_training.py /app/
COPY plugins/cd4ml/model_training/requirements-model.txt /app/
COPY data/processed /app/data/processed
COPY models /app/models

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements-model.txt

ENV MODEL_OUTPUT_DIR=/app/models
ENV DATA_INPUT_DIR=/app/data/processed

CMD ["python", "run_model_training.py"]