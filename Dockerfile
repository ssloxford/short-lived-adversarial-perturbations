FROM tensorflow/tensorflow:1.15.3-gpu-py3-jupyter
WORKDIR /home

# expose this port for jupyter notebook
EXPOSE 5749

# avoid questions when installing packages in apt-get
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update
RUN apt-get -y install python-opencv
RUN apt-get -y install python3-tk
RUN apt-get -y install git wget
RUN apt-get -y install ffmpeg
RUN apt-get -y install imagemagick
RUN apt-get -y install v4l-utils
RUN apt-get -y install libcurl4-openssl-dev libssl-dev

# install pip requirements
COPY requirements.txt /home/requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install --no-deps saliency==0.0.5

# external dependencies
# 1. we use a few utils from the wizyoung/YOLOv3_TensorFlow repository
# clone from my fork https://github.com/giuliolovisotto/YOLOv3_TensorFlow.git
RUN git clone https://github.com/giuliolovisotto/YOLOv3_TensorFlow.git
ENV PYTHONPATH="/home/:/home/code/:/home/YOLOv3_TensorFlow/:$PYTHONPATH"

# overrides entry point from base image
WORKDIR /home
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.password='argon2:\$argon2id\$v=19\$m=10240,t=10,p=8\$To1X6n4bCpn7nf0W0yShsQ\$eoTYo0cRJVbGJIccVkOBfQ'">>/root/.jupyter/jupyter_notebook_config.py
CMD nohup jupyter notebook --port=5749 --allow-root --ip=0.0.0.0 & bash

