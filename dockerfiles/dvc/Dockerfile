FROM python:3.11-slim

WORKDIR /workspace

# install git and curl
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# upgrade pip
RUN pip install --upgrade pip

# install dvc with s3 support
RUN pip install dvc[s3]

# pull the data from s3
CMD ["dvc", "pull"]
