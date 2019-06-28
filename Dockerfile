FROM pypy:3.6

RUN apt-get update && apt-get -y install libgeos-dev build-essential swig cmake lsb-release pypy-setuptools

# Ortools
RUN wget https://github.com/google/or-tools/archive/v7.1.zip && \
    unzip v7.1.zip

RUN apt-get install -y python3-pip

RUN cd or-tools-7.1 && \
    make -j 4 third_party

RUN cd or-tools-7.1 && \
    make -j 4 cc

RUN echo "deb http://ftp.debian.org/debian testing main" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -t testing -y g++

COPY requirements.txt or-tools-7.1/
RUN pypy3 -m pip install -r or-tools-7.1/requirements.txt

RUN cd or-tools-7.1/ && mkdir /usr/local/site-packages/ortools_space_planner_pypy && \
    rm -rf /v7.1.zip

COPY requirements.txt .
RUN pip install -r requirements.txt
#RUN apt-get -y install vim

COPY selection.xml /or-tools-7.1
RUN cd or-tools-7.1 && \
    make -j 4 cc

RUN cd /or-tools-7.1 && genreflex ortools/constraint_solver/constraint_solver.h --cxxflags -Idependencies/sources/abseil-cpp-bf29470/ -I ortools/gen/ -Idependencies/install/include/ -o ortools_pypy.cpp -s selection.xml --rootmap=ortools_pypy.rootmap --rootmap-lib=libOrtoolsPypyDict.so && \
    g++ -std=c++11 -fPIC -rdynamic -O2 -shared `genreflex --cppflags` ortools_pypy.cpp -o libOrtoolsPypyDict.so  -Idependencies/sources/abseil-cpp-bf29470/ -I ortools/gen/ -Idependencies/install/include -I. -L ./lib -lortools -L/or-tools-7.1/lib -L/usr/local/site-packages/cppyy_backend/lib/ -lCling && \
    cp /or-tools-7.1/libOrtoolsPypyDict.so /or-tools-7.1/ortools_pypy.rootmap /usr/local/site-packages/cppyy_backend/lib/ && \
    cp -rp /or-tools-7.1//dependencies/install/lib/* /usr/lib/ && \
    cp /usr/local/site-packages/cppyy_backend/lib/libCling.so /usr/lib/ && \
    cp -pr  /or-tools-7.1/dependencies/install/include/* /usr/local/site-packages/cppyy_backend/include/ && \
    cp /or-tools-7.1/lib/libortools.so /usr/lib && \
    cp -pr /or-tools-7.1/ortools/gen/* /usr/local/site-packages/cppyy_backend/include/ && \
    cp /or-tools-7.1/ortools_pypy_rdict.pcm /usr/local/site-packages/cppyy_backend/lib/

COPY ortools_space_planner_pypy/__init__.py /usr/local/site-packages/ortools_space_planner_pypy/
