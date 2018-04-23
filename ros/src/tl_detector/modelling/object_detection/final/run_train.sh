#!/bin/bash

DIRNAME=$(dirname "$0")

cd ${DIRNAME}

# Bosch training
#python train.py --logtostderr --train_dir=./models/train/faster_rcnn_inception_v2 --pipeline_config_path=./faster_rcnn_inception_v2_bosch.config
#python train.py --logtostderr --train_dir=./models/train/ssd_mobilenet_v2 --pipeline_config_path=./ssd_mobilenet_v2_bosch.config

# Udacity real
mkdir -p ./models/train/faster_rcnn_inception_v2_udacity_real && rm -rf ./models/train/faster_rcnn_inception_v2_udacity_real/* && python train.py --logtostderr --train_dir=./models/train/faster_rcnn_inception_v2_udacity_real --pipeline_config_path=./faster_rcnn_inception_v2_udacity_real.config
mkdir -p ./models/train/ssd_mobilenet_v2_udacity_real && rm -rf ./models/train/ssd_mobilenet_v2_udacity_real/* && python train.py --logtostderr --train_dir=./models/train/ssd_mobilenet_v2_udacity_real --pipeline_config_path=./ssd_mobilenet_v2_udacity_real.config

# Udacity sim
mkdir -p ./models/train/faster_rcnn_inception_v2_udacity_sim && rm -rf ./models/train/faster_rcnn_inception_v2_udacity_sim/* && python train.py --logtostderr --train_dir=./models/train/faster_rcnn_inception_v2_udacity_sim --pipeline_config_path=./faster_rcnn_inception_v2_udacity_sim.config
mkdir -p ./models/train/ssd_mobilenet_v2_udacity_sim && rm -rf ./models/train/ssd_mobilenet_v2_udacity_sim/* && python train.py --logtostderr --train_dir=./models/train/ssd_mobilenet_v2_udacity_sim --pipeline_config_path=./ssd_mobilenet_v2_udacity_sim.config
