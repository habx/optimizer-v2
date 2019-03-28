# Execute with:
# =============
# docker build . -t worker-optimizer-v2 && docker run worker-optimizer-v2

FROM python:3.6
WORKDIR /app
CMD bin/worker.py
COPY . /app
RUN pip install -r requirements.txt
