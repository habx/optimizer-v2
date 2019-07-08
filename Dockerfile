FROM habx/ds-opt-pypy:latest
COPY . .
RUN    pip3 install -r requirements.txt
