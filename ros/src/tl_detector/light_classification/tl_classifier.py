from styx_msgs.msg import TrafficLight
import sys
import os
import collections
import cv2
import numpy as np
import tensorflow as tf
from utilities import label_map_util
from utilities import visualization_utils as viz


class TLClassifier(object):
    def __init__(self, threshold=0.5):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.threshold = threshold

        self.model_name = 'ssd_mobilenet_v1_udacity_all_1.4'
        self.model_path = dir_path + '/models/' + self.model_name + \
                                     '/frozen_inference_graph.pb'
        self.labels_name = 'udacity_all'
        self.labels_path = dir_path + '/labels/' + self.labels_name + ".pbtxt"

        # add classification boxes to image
        label_map = label_map_util.load_labelmap(self.labels_path)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=3, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        self.detection_graph = self.load_tf_model()

        self.classified_image = None

    def get_classification(self, image_bgr8):
        """Determines the color of the traffic light in the image

        Args:
            image_bgr8 (cv::Mat): image containing the traffic light (assumed to be a bgr8 image)

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_rgb_np = cv2.cvtColor(image_bgr8, cv2.COLOR_BGR2RGB)
        output_dict = self.run_inference_for_single_image(image_rgb_np)

        viz.visualize_boxes_and_labels_on_image_array(image_rgb_np,
                                                      output_dict['detection_boxes'],
                                                      output_dict['detection_classes'],
                                                      output_dict['detection_scores'],
                                                      self.category_index,
                                                      use_normalized_coordinates=True)
        self.classified_image = image_rgb_np
        
        # Get the highest classifying score. The classifier will always return n number of classifications.
        highest_score_pos = np.argmax(output_dict['detection_scores'])
        if output_dict['detection_scores'][highest_score_pos] < self.threshold:
            return TrafficLight.UNKNOWN
        
        else:
            detection_class = output_dict['detection_classes'][highest_score_pos]
            if detection_class == 1:
                return TrafficLight.GREEN
            elif detection_class == 2:
                return TrafficLight.RED
            elif detection_class == 3:
                return TrafficLight.YELLOW
        
        return TrafficLight.UNKNOWN

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
                    detection_masks_reframed = reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict


def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width):
    """Transforms the box masks back to full image masks.

    Embeds masks in bounding boxes of larger masks whose shapes correspond to
    image shape.

    Args:
    box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
    boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
           corners. Row i contains [ymin, xmin, ymax, xmax] of the box
           corresponding to mask i. Note that the box corners are in
           normalized coordinates.
    image_height: Image height. The output mask will have the same height as
                  the image height.
    image_width: Image width. The output mask will have the same width as the
                 image width.

    Returns:
    A tf.float32 tensor of size [num_masks, image_height, image_width].
    """
    # TODO(rathodv): Make this a public function.
    def transform_boxes_relative_to_boxes(boxes, reference_boxes):
        boxes = tf.reshape(boxes, [-1, 2, 2])
        min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
        max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
        transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
        return tf.reshape(transformed_boxes, [-1, 4])

    box_masks = tf.expand_dims(box_masks, axis=3)
    num_boxes = tf.shape(box_masks)[0]
    unit_boxes = tf.concat(
      [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
    reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
    image_masks = tf.image.crop_and_resize(image=box_masks,
                                         boxes=reverse_boxes,
                                         box_ind=tf.range(num_boxes),
                                         crop_size=[image_height, image_width],
                                         extrapolation_value=0.0)
    return tf.squeeze(image_masks, axis=3)


