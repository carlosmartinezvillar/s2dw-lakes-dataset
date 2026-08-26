FROM nvidia/cuda:12.4.1-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
	python3.10 \
	python3-pip \
	git \
	zip \
	unzip

RUN pip install -U pip \
	&& pip install torch torchvision numpy pillow pyyaml \
	nvitop torchprofile

