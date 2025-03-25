# data_processing/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# copy requirements.txt
# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy all files
COPY . .

# NLTK-Downloads, if needed
RUN python -m nltk.downloader punkt stopwords wordnet

# run the preprocessing script
CMD ["python", "run_preprocessing.py"]
