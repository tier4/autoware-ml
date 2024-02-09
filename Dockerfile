ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ARG MMCV="2.0.0rc4"
ARG MMDET="3.3.0"
ARG MMDET3D="1.4.0"

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# (Optional, use Mirror to speed up downloads)
# RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
#    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMEngine, MMCV and MMDetection
RUN pip install openmim && \
    mim install "mmengine" "mmcv>=2.0.0rc4" "mmdet>=3.0.0"

# Install MMDetection3D
RUN mim install mmcv==${MMCV}
RUN mim install mmdet==${MMDET}
RUN mim install mmdet3d==${MMDET3D}

#RUN conda clean --all \
#   && git clone https://github.com/open-mmlab/mmdetection3d.git -b v1.4.0 /mmdetection3d \
#   && cd /mmdetection3d \
#   && pip install --no-cache-dir -e .
# WORKDIR /mmdetection3d

ARG WORKSPACE
WORKDIR /workspace

COPY autoware_ml autoware_ml
COPY configs configs
COPY projects projects
COPY tools tools
COPY setup.py setup.py
COPY README.md README.md

RUN pip install --no-cache-dir -e .

# COPY third_party/mmdetection3d third_party/mmdetection3d 
# RUN conda clean --all && cd third_party/mmdetection3d && pip install --no-cache-dir -e .

