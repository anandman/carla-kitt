#!/bin/bash

echo "Please make sure that you have the correct python environment installed and activated BEFORE you run this."

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

aws s3 sync s3://udacity-kitt-training-datasets ./$SCRIPTPATH/data

wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_kitti_2018_01_28.tar.gz $SCRIPTPATH/faster_rcnn_resnet101_kitti_2018_01_28.tar.gz
tar xzvf $SCRIPTPATH/faster_rcnn_resnet101_kitti_2018_01_28.tar.gz .

cd $SCRIPTPATH/cocoapi/PythonAPI
make

# tensorflow_path=$(pip show tensorflow | grep Location | cut -d' ' -f 2)