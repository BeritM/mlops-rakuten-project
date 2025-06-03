FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

COPY dockerfiles/model_training/requirements-model.txt ./requirements-model.txt
RUN pip install --no-cache-dir -r requirements-model.txt
RUN pip install --no-cache-dir "dvc[s3]"  

COPY . .

ENV MODEL_OUTPUT_DIR=/app/models
ENV DATA_INPUT_DIR=/app/data/processed

CMD ["python", "plugins/cd4ml/model_training/run_model_training.py"]
