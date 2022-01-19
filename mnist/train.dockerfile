# Base image

FROM python:3.8-slim

# install python
# RUN apt-get update && \
# apt-get install --no-install-recommends -y build essential gcc && \
# apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

WORKDIR /
# Installs requirements and disables cache to minimize image size
RUN pip install -r requirements.txt --no-cache-dir 

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
