#!/bin/bash

DIRNAME=$(dirname "$0")

cd ${DIRNAME}

python train.py --logtostderr --train_dir=./models/train/faster_rcnn_inception_v2 --pipeline_config_path=./faster_rcnn_inception_v2_bosch.config

python train.py --logtostderr --train_dir=./models/train/ssd_mobilenet_v2 --pipeline_config_path=./ssd_mobilenet_v2_bosch.config