#!/bin/bash

echo "Please make sure that you have the correct python environment installed and activated BEFORE you run this."

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# aws s3 sync s3://udacity-kitt-training-datasets ./$SCRIPTPATH/data

#wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz $SCRIPTPATH/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
#tar xzvf $SCRIPTPATH/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz .

#cd $SCRIPTPATH/cocoapi/PythonAPI
#make

site_packages_path=$(pip show tensorflow | grep Location | cut -d' ' -f 2)

# Need to follow https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e
#cd $SCRIPTPATH/tensorflow/research/
#protoc object_detection/protos/*.proto --python_out=.

#echo $SCRIPTPATH/tensorflow/research/slim > $site_packages_path/slim.pth
#echo $SCRIPTPATH/tensorflow/research > $site_packages_path/research.pth
