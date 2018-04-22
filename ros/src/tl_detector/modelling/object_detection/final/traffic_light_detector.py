import sys
sys.path.insert(1, "../../tensorflow/research")
sys.path.insert(1, "../../tensorflow/research/slim")
sys.path.append("..")
sys.path.append("../../tensorflow/research")
sys.path.append("../../tensorflow/research/slim")

import glob
import numpy as np
import os
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

dir_path = os.path.dirname(os.path.realpath(__file__))

class TLDetector(object):
    def __init__(self, model_name, labels_name):
        self.model_name = model_name
        self.model_path = dir_path + '/models/' + self.model_name + \
                                     '/frozen_inference_graph.pb'
        self.labels_name = labels_name
        self.labels_path = dir_path + '/labels/' + self.labels_name + ".pbtxt"

        self.detection_graph = self.load_tf_model()

    def load_tf_model(self):
        detection_graph = tf.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        return detection_graph

    def run_inference_for_single_image(self, image):
        '''
        Call this function with a numpy image array and a graph structure
        '''
        graph = self.detection_graph
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(
                        tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                            real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                            real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                        feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def retrieve_traffic_lights(self, output_dict):
        '''
        norm_boxes: a 2 dimensional numpy array of[N, 4]: (ymin, xmin, ymax, xmax).
            The coordinates are in normalized format between[0, 1].
        '''

        traffic_light_idx = np.argwhere(output_dict['detection_classes'] == 10)

        norm_boxes = output_dict['detection_boxes'][traffic_light_idx]
        scores = output_dict['detection_scores'][traffic_light_idx]

        return norm_boxes, scores

    def get_truncated_model_for_training(self):
        graph = self.detection_graph
        with graph.as_default():
            ([print(n.name) for n in tf.get_default_graph().as_graph_def().node])
            

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


if __name__ == "__main__":
    from PIL import Image

    PATH_TO_TEST_IMAGES_DIR = '../../../data/udacity'
    TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR + "/*.jpg")

    MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
    LABEL_NAME = 'mscoco_label_map'

    tl_detector = TLDetector(MODEL_NAME, LABEL_NAME)

    for image_path in TEST_IMAGE_PATHS:
        print(image_path)
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = tl_detector.run_inference_for_single_image(image_np)
        norm_boxes, scores = tl_detector.retrieve_traffic_lights(output_dict)

        print("norm_boxes (ymin, xmin, ymax, xmax): %s", norm_boxes)
        print("scores: %s", scores)
