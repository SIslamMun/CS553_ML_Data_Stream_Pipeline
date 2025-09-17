# Getting Started


# CS553_ML_Data_Stream_Pipeline
ML data streaming pipeline using Apache Flink
This project is a simple implementation of an ML model inference system. It uses the Apache Flink framework for processing data streams.

## Install Dependencies
Assuming you have <a href="https://docs.anaconda.com/anaconda/install/">Anaconda</a>.
Now, Follow the below steps:

- Step 1: conda create --name cs553
- Step 2: conda activate cs553
- Step 3: pip install -r requirements.txt 


















or 


 Assuming you have docker and Cuda installed.
 else,
## Docker Installation 
### open a terminal
 ```
 curl -fsSL https://get.docker.com -o get-docker.sh
 ```
 ```
 sudo sh get-docker.sh
 ```
 ```
 sudo systemctl restart docker
 ```

 Now, Follow the below steps:

## STEP 1:
 ```
 cd [project folder path]/docker
 ```
 ```
 sudo docker build -t [docker image name] .
 ```
## STEP 2:
 ```
 sudo docker images
 ```
 ```
 sudo docker run -it [docker image id] bash
 ```

## STEP 3:
 ```
 apt-get update
 ```
 ```
 apt-get install -y libopencv-dev
 ```

## STEP 4:
 ### open another terminal
 ```
 sudo docker images
 ```
 ```
 sudo docker ps
 ```
 ```
 sudo docker commit [container id from previous command] [docker image name:tag name]
 ```

## STEP 5 (gpu device only else skip):
 ```
 distribution=ubuntu20.04
 ```
 ```
 curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
 ```
 ```
 curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
 ```
 ```
 sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
 ```
 ```
 sudo docker pull nvidia/cuda:11.2.0-base-ubuntu20.04
 ```
 ```
 docker run --rm --gpus all nvidia/cuda:11.2.0-base-ubuntu20.04 nvidia-smi
 ```

## STEP 6:
 ### For gpu:
  ```
  sudo docker run -e DEVICE="cuda" -v [project folder path]:/home/workspace --gpus all -it [image id] --network host bash
  ```

