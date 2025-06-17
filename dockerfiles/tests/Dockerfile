# Use a slim base image
FROM python:3.11-slim

# Set working directory for the build context initially
WORKDIR /app

# Install system dependencies, including Git, Curl, and gosu
RUN apt-get update \
  && apt-get install -y --no-install-recommends git curl gosu \
  && rm -rf /var/lib/apt/lists/*

# Git safe directory configuration (important for DVC related tests if applicable)
#RUN git config --global --add safe.directory /app && \
#    git config --global --add safe.directory '*'

# Copy requirements & install Python dependencies
COPY dockerfiles/tests/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for testing
RUN pip install --no-cache-dir requests

# Copy the entrypoint script and make it executable
COPY dockerfiles/tests/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Copy entire project
COPY . .

# Set the entrypoint to our script, and the default command
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["pytest", "plugins/cd4ml/tests/test_predict_service.py", "-v", "-rA"]