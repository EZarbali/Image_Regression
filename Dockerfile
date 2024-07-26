FROM nvcr.io/nvidia/pytorch:23.12-py3
#FROM python:3.10
#FROM pytorch/pytorch

ENV DEBIAN_FRONTEND=noninteractive
ENV CRASH_HANDLER=FALSE

WORKDIR /workspace/app

RUN pip --no-cache-dir install Cython

RUN pip install --upgrade pip

COPY . /app 

COPY ./requirements.txt /workspace

RUN pip3 install --upgrade pip && pip3 --no-cache-dir install -r /workspace/requirements.txt


RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
#RUN pip --no-cache-dir install -r /workspace/requirements.txt

EXPOSE 80

ENV NAME World

WORKDIR /app 
