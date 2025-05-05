FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

COPY dockerfiles/model_validation/requirements-model.txt .  
RUN pip install --no-cache-dir -r requirements-model.txt

COPY plugins/cd4ml/model_validation /app/plugins/cd4ml/model_validation

ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV MODEL_OUTPUT_DIR=/app/models
ENV DATA_INPUT_DIR=/app/data/processed

CMD ["python", "plugins/cd4ml/model_validation/run_model_validation.py"]