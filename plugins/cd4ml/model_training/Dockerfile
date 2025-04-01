FROM python:3.10-slim

WORKDIR /app

COPY model_training.py /app/
COPY run_model_training.py /app/
COPY requirements-model.txt /app/
COPY ../../../data/processed /app/data/processed
COPY ../../../models /app/models

RUN pip install --no-cache-dir -r requirements-model.txt

ENV MODEL_OUTPUT_DIR=/app/models
ENV DATA_INPUT_DIR=/app/data/processed

CMD ["python", "run_model_training.py"]