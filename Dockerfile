# Execute with:
# =============
# docker build . -t test && docker run test

FROM python:3.6
COPY . /app
RUN cd /app && ./run_tests.sh
