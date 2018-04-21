#!/bin/bash

DIRNAME=$(dirname "$0")

cd ${DIRNAME}

python train.py --logtostderr --train_dir=./models/train --pipeline_config_path=./ssd_inception_v2_coco.config

python train.py --logtostderr --train_dir=./models/train --pipeline_config_path=./ssd_mobilenet_v2_coco.config