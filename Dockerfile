FROM pypy:3.6-stretch

RUN echo "deb http://ftp.debian.org/debian testing main" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -t testing -y g++

RUN wget https://habx-artifacts.s3-eu-west-1.amazonaws.com/ortools-cppyy.tar.gz && \
    tar xvzf ortools-cppyy.tar.gz && \
    mkdir -p /usr/local/site-packages/cppyy_backend/{lib,include} && \
    cp -pr ortools-cppyy/lib/* /usr/local/site-packages/cppyy_backend/lib/ && \
    cp -pr ortools-cppyy/lib/* /usr/lib/ && \
    cp -pr ortools-cppyy/include/* /usr/local/site-packages/cppyy_backend/include/

COPY . /app

# GCC 8+ is needed to build cppyy
RUN pip3 install -r /app/requirements.txt

RUN cp -pr /app/ortools_space_planner_pypy/ /usr/local/site-packages/

ENV CPPYY_DISABLE_FASTPATH=1
CMD ["bin/worker.py"]
