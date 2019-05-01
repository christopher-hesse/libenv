FROM ubuntu:bionic-20190122
RUN apt-get update
RUN apt-get install --yes curl build-essential

# python
RUN curl -O https://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh
RUN sh Miniconda3-4.6.14-Linux-x86_64.sh -b
ENV PATH=/root/miniconda3/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ADD env.yaml .
RUN conda env update --name base --file env.yaml

# go
# conda-forge version of go seems to have some sort of issue
# possibly https://github.com/golang/go/issues/24068
RUN curl -O https://dl.google.com/go/go1.12.4.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf go1.12.4.linux-amd64.tar.gz
ENV PATH=/usr/local/go/bin:$PATH