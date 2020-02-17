# Execute with:
# =============
# docker build . -t worker-optimizer-v2 && docker run worker-optimizer-v2
FROM python:3.7-slim
ARG GEMFURY_READ_TOKEN
ARG GEMFURY_ORGA

RUN apt-get update && apt-get install libgoogle-perftools4 git gcc -y
# If we want to make it the default behavior, we can to uncomment this:
# ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
WORKDIR /app
COPY . /app
CMD ["bin/worker.sh"]
RUN pip install --upgrade pip && pip install -r requirements.txt
