# Execute with:
# =============
# docker build . -t test && docker run test

FROM python:3.6
COPY . /app
CMD ["bin/worker.py"]
