# Create container with
# docker run -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1 pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ARG PYTORCH="2.0.0"
ARG CUDA="11.7"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

RUN apt update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 nano mc glances vim git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install cython
RUN pip install packaging

# Apex
RUN git clone https://github.com/NVIDIA/apex
RUN conda remove cuda-nvcc -y
RUN conda install -c "nvidia/label/cuda-11.7.1" cuda-nvcc -y
RUN pip install -v --disable-pip-version-check --no-cache-dir ./

RUN apt-get update -y
RUN apt-get install build-essential cmake -y
RUN apt-get install libopenblas-dev liblapack-dev -y
RUN apt-get install libx11-dev libgtk-3-dev -y
RUN pip install dlib==19.24.1
RUN pip install facenet-pytorch==2.5.3
RUN pip install albumentations==1.3.0 timm==0.9.2 pytorch_toolbelt==0.6.3 pandas==2.0.1 wandb vessl

# download pretraned Imagenet models
RUN apt install wget
RUN wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth -P /root/.cache/torch/hub/checkpoints/
RUN wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ns-6f26d0cf.pth -P /root/.cache/torch/hub/checkpoints/

# Setting the working directory
WORKDIR /workspace

# Copying the required codebase
COPY . /workspace

RUN chmod 777 preprocess_data.sh
RUN chmod 777 train.sh
RUN chmod 777 predict_submission.sh

ENV PYTHONPATH=.

CMD ["/bin/bash"]
