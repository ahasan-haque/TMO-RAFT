# Use an official Python runtime as the base image

FROM python:3.9-slim

# Set the working directory
WORKDIR /app
COPY . /app

WORKDIR /app

# # need to run on /bin/bash mode
RUN apt update || true \
       && apt install -y software-properties-common \
       && add-apt-repository ppa:deadsnakes/ppa \
       && apt install -y unzip git vim build-essential \
       zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
       libssl-dev libreadline-dev libffi-dev wget build-essential \
       cmake git libjpeg-dev libtiff5-dev libpng-dev libavcodec-dev \
       libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev \
       libv4l-dev v4l-utils libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
       libgtk2.0-dev mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev \
       libatlas-base-dev gfortran libeigen3-dev python3-dev python3-numpy 

RUN python3.9 -m pip install --upgrade pip && python3.9 -m pip install -r requirements.txt

RUN    git clone --branch 4.5.1 https://github.com/opencv/opencv.git --single-branch  \
    && git clone --branch 4.5.1 https://github.com/opencv/opencv_contrib.git --single-branch \
    && cd /app/opencv_contrib \
    && git checkout 4.5.1 \
    && cd /app/opencv \
    && git checkout 4.5.1 \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE \
       -D CMAKE_INSTALL_PREFIX=/usr/local \
       -D INSTALL_C_EXAMPLES=ON \
       -D WITH_CUDA=ON \
       -D INSTALL_PYTHON_EXAMPLES=ON \
       -D WITH_TBB=ON \
       -D WITH_V4L=ON \
       -D WITH_OPENGL=ON \
       -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
       -D BUILD_EXAMPLES=ON .. && \
       make -j16 && \
       make install && \
       rm -rf opencv_contrib opencv

#WORKDIR /app
#RUN wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip && unzip models.zip

#RUN rm -rf opencv opencv_contrib models.zip
#RUN echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf
#RUN ldconfig
#RUN python3.7 -m pip install -r requirements.txt

# docker run --gpus all -it -v  c:\Users\ge79pih\color\color:/app/color raft_optical_map_running /bin/bash
#docker run -e MODEL_PATH=models/my_model.pth \
#           -e DATASET_PATH=/datasets \
#           -e OUTPUT_PATH=/output \
#           -v c:\Users\ge79pih\color\color\custom_png_data:/datasets \
#           -v c:\Users\ge79pih\color\color\demo_output:/output \
#           --gpus all
#           raft

#CMD sh -c 'python3.8 evaluate.py --model=$MODEL_PATH --dataset-path=$DATASET_PATH --output-path=$OUTPUT_PATH'

CMD sh -c 'python3.9 run.py --test --model=$MODEL_PATH --dataset-path=$DATASET_PATH --output-path=$OUTPUT_PATH'