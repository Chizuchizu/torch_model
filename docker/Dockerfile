FROM nvidia/cuda:11.7.0-runtime-ubuntu22.04

RUN apt-get update
RUN apt-get install -y python3 python3-pip

# pytorch
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113


RUN mkdir /api
ADD . /api
WORKDIR /api
RUN pip3 install -r ./requirements.txt
