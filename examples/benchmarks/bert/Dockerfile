FROM docker.io/pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# 已内置：
# - Python 3.8+
# - pip
# - CUDA 11.7
# - cuDNN 8
# - PyTorch、Torchvision、Torchaudio

# 安装其他依赖
WORKDIR /workspace
COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]