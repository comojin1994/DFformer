#!/bin/bash

directory=$(cd ../ && pwd)

docker run --restart always \
-d \
-it \
--name dfformer \
--runtime nvidia \
--ipc=host \
--gpus all \
-v ${directory}:/opt/pytorch \
comojin1994/cu11.2-ubuntu-18.04-pytorch-1.10.0:0.8 \
/bin/bash;