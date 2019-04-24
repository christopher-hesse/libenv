FROM ubuntu:bionic-20190122
RUN apt-get update
RUN apt-get install --yes curl build-essential cmake

# python
RUN curl -O https://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh
RUN sh Miniconda3-4.6.14-Linux-x86_64.sh -b
ENV PATH=/root/miniconda3/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN conda install --yes python==3.7.3

# go
RUN curl -O https://dl.google.com/go/go1.12.4.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf go1.12.4.linux-amd64.tar.gz
ENV PATH=/usr/local/go/bin:$PATH