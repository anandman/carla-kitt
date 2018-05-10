from styx_msgs.msg import TrafficLight
import tensorflow as tf
import os
import cv2
import numpy as np
import rospy

class TLClassifier(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.detection_model_path = rospy.get_param("/detection_model_path")

        pass

    def get_classification(self, image_bgr8):
        """Determines the color of the traffic light in the image

        Args:
            image_bgr8 (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_rgb_np = cv2.cvtColor(image_bgr8, cv2.COLOR_BGR2RGB)
        output_dict = self.tl.run_inference_for_single_image(image_rgb_np)

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
