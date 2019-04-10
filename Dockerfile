# Execute with:
# =============
# docker build . -t worker-optimizer-v2 && docker run worker-optimizer-v2

FROM python:3.7
WORKDIR /app
CMD bin/worker.py
COPY . /app
CMD ["bin/worker.py"]
RUN pip install --upgrade pip && pip install -r requirements.txt
