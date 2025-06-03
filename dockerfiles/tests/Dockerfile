FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy & install requirements
COPY dockerfiles/tests/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for testing
RUN pip install --no-cache-dir requests

# Copy entire project
COPY . .

# Command
CMD ["pytest", "plugins/cd4ml/tests/test_predict_service.py", "-v", "-rA"]