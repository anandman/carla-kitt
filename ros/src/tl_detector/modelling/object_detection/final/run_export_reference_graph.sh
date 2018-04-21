#!/bin/bash

python export_reference_graph.py --input_type image_tensor --pipeline_config_path ./models/train/faster_rcnn_inception_v2/pipeline.config --trained_checkpoint_prefix ./models/train/faster_rcnn_inception_v2/model.ckpt-200 --output_directory ./models/faster_rcnn_inception_v2_bosch
python export_reference_graph.py --input_type image_tensor --pipeline_config_path ./models/train/ssd_mobilenet_v2/pipeline.config --trained_checkpoint_prefix ./models/train/ssd_mobilenet_v2/model.ckpt-200 --output_directory ./models/ssd_mobilenet_v2_bosch
