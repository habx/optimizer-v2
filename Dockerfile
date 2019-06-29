FROM pypy:3.6-slim

RUN echo "deb http://ftp.debian.org/debian testing main" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -t testing -y g++

# GCC 8+ is needed to build cppyy
RUN pip3 install -r requirements.txt

COPY . /app
RUN cp -pr /app/ortools_space_planner_pypy/ /usr/local/site-packages/ && \
    wget https://habx-artifacts.s3-eu-west-1.amazonaws.com/ortools-cppyy.tar.gz && \
    tar xvzf ortools-cppyy.tar.gz && \
    cp -pr ortools-cppyy/lib/* /usr/local/site-packages/cppyy_backend/lib/ && \
    cp -pr ortools-cppyy/lib/* /usr/lib/ && \
    cp -pr ortools-cppyy/include/* /usr/local/site-packages/cppyy_backend/include/


ENV CPPYY_DISABLE_FASTPATH=1
CMD ["bin/worker.py"]
