# Convert the raw training data into record files that can be used for training
import sys
import os
# Import all external libs
DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIRNAME + "/../lib")

import tensorflow as tf
import yaml
from object_detection.utils import dataset_util
from data_conversion import bosch, udacity
import glob

def create_tf_example(filename, img_annotations, label_dict):
    _, imgfmt = os.path.splitext(filename)

    # Filename of the image. Empty if image is not from file
    encoded_filename = filename.encode()

    with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_image = fid.read()

    image_format = imgfmt.encode()

    # List of normalized left x coordinates in bounding box (1 per box)
    xmins = []
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    # List of normalized top y coordinates in bounding box (1 per box)
    ymins = []
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    for img_annotation in img_annotations:
        xmins.append(float(img_annotation['x_min']) / img_annotation['im_width'])
        xmaxs.append(float(img_annotation['x_max']) / img_annotation['im_width'])
        ymins.append(float(img_annotation['y_min']) / img_annotation['im_height'])
        ymaxs.append(float(img_annotation['y_max']) / img_annotation['im_height'])
        classes_text.append(img_annotation['class'].encode())
        classes.append(int(label_dict[img_annotation['class']]))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(img_annotation['im_height']),
        'image/width': dataset_util.int64_feature(img_annotation['im_width']),
        'image/filename': dataset_util.bytes_feature(encoded_filename),
        'image/source_id': dataset_util.bytes_feature(encoded_filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example

converter_mapping = {
    "bosch_test.yaml": bosch.Converter,
    "bosch_train.yaml": bosch.Converter,
    "udacity_real.yaml": udacity.Converter,
    "udacity_sim.yaml": udacity.Converter,
}

default_tf_records = {
    "bosch_test.record": ["bosch_test.yaml"],
    "udacity_real.record": ["udacity_real.yaml"],
    "udacity_sim.record": ["udacity_sim.yaml"],
    "udacity_all.record": ["udacity_real.yaml", "udacity_sim.yaml"],
}

def generate_labels(yaml_files):
    labels_set = set()
    for yaml_file in yaml_files:
        converter = converter_mapping[yaml_file](yaml_file)
        image_annotations = converter.get_annotations()
        for image_annotation in image_annotations:
            labels_set.add(image_annotation["class"])

    items_pbtxt = []
    label_dict = {}
    idx = 1
    for label in labels_set:
        items_pbtxt.append("""item {
    id: %d
    name: '%s'
}""" % idx, label)
        label_dict[label] = idx
        idx += 1

    return "\n".join(items_pbtxt), label_dict


def run(tf_records = None):
    tf_records = default_tf_records
    for tf_record in tf_records:
        yaml_files = tf_records[tf_record]

        for idx in range(len(yaml_files)):
            yaml_files[idx] = DIRNAME + "/data/yaml/" + yaml_files[idx]

        # Generate labels
        generated_pbtxt, label_dict = generate_labels(yaml_files)
        # save labels
        record_name, _ = os.path.splitext(tf_record)
        label_path = DIRNAME + "/data/converted/labels/" + record_name + ".pbtxt"
        with open(label_path, "w") as text_file:
            text_file.write(generated_pbtxt)

        tf_record_path = DIRNAME + "/data/records/" + tf_record
        writer = tf.python_io.TFRecordWriter(tf_record_path)
        for yaml_file in yaml_files:
            converter = converter_mapping[yaml_file](yaml_file)
            abs_paths = converter.get_absolute_paths()
            image_annotations = converter.get_annotations()

            for img_idx in range(len(abs_paths)):
                tf_example = create_tf_example(abs_paths[img_idx], image_annotations[img_idx], label_dict)
                writer.write(tf_example.SerializeToString())

        writer.close()

if __name__ == "__main__":
    run()
