FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

LABEL maintainer="Evgeny Nizhibitsky <nizhib@github.com>"

# Install ubuntu base packages

RUN apt-get update
RUN apt-get install -y python3-pip

# Install python packages

WORKDIR /opt/portrait-demo/backend/
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy project files

COPY . .

# Run the service

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
CMD python3 -u app.py
