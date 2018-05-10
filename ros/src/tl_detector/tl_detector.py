#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import numpy as np
from scipy.spatial import KDTree
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

STATE_COUNT_THRESHOLD = 3


def light_state_text(state):
    """Returns the color state of the traffic light

    Args:
        state (TrafficLight.state): light state to report

    Returns:
        str: color of the traffic light state(specified in styx_msgs/TrafficLight)

    """
    if state == TrafficLight.GREEN:
        return "GREEN"
    elif state == TrafficLight.YELLOW:
        return "YELLOW"
    elif state == TrafficLight.RED:
        return "RED"
    else:
        return "UNKNOWN"


def distance(p1, p2):
    x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
    return math.sqrt(x * x + y * y + z * z)


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector', log_level=rospy.INFO)

        # do we want to use ground truth or detected traffic lights?
        self.use_ground_truth = rospy.get_param("/use_ground_truth")

        # do we want to save camera images for training/testing purposes
        self.save_images = rospy.get_param("/save_images")

        # distance ahead of vehicle to check for lights
        self.check_light_distance = rospy.get_param("/check_light_distance")

        self.pose = None

        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None

        self.has_image = False
        self.camera_image = None

        self.lights = []
        self.lights_2d = None
        self.light_tree = None

        self.next_traffic_light_dist = None

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and 
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.light_classifier_pub = rospy.Publisher('/light_classifier', Image, queue_size=1)
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.logdebug("TL_DETECTOR: starting spin...")

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in
                                 waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights
        if not self.lights_2d:
            self.lights_2d = [[light.pose.pose.position.x, light.pose.pose.position.y] for light in self.lights]
            self.light_tree = KDTree(self.lights_2d)

        # this allows for us not to have to turn camera on to use ground truth
        if self.use_ground_truth and self.waypoint_tree:
            light_wp, state = self.process_traffic_lights()
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                light_wp = light_wp if state == TrafficLight.RED or state == TrafficLight.YELLOW else -1
                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            self.state_count += 1

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED or state == TrafficLight.YELLOW else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose, use_lights=False):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem

        Args:
            pose (Pose): position to match a waypoint to
            use_lights: specify whether to search through just traffic lights or all waypoints

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        x = pose.position.x
        y = pose.position.y

        if use_lights:
            waypoints_2d, waypoint_tree = self.lights_2d, self.light_tree
        else:
            waypoints_2d, waypoint_tree = self.waypoints_2d, self.waypoint_tree

        closest_idx = waypoint_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind vehicle
        closest_coord = waypoints_2d[closest_idx]
        prev_coord = waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest coord
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(waypoints_2d)

        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        state = TrafficLight.UNKNOWN

        if self.use_ground_truth:
            rospy.logdebug("TL_DETECTOR: using ground truth light state: %s", light_state_text(light.state))
            # Return the classification of the traffic light based on the simulation /vehicle/traffic_lights
            return light.state

        if not self.has_image:
            # self.prev_light_loc = None
            return state

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # x, y = self.project_to_image_plane(light.pose.pose.position)

        # Get classification
        state = self.light_classifier.get_classification(cv_image)
        rospy.logdebug("TL_DETECTOR: detected light state: %s", light_state_text(state))

        self.light_classifier_pub.publish(self.bridge.cv2_to_imgmsg(self.light_classifier.classified_image, "rgb8"))

        if self.save_images:
            # Save camera images alongside distance and status metadata for training and testing
            light_timestamp_epoch_secs = light.pose.header.stamp.secs
            light_timestamp_epoch_nsecs = light.pose.header.stamp.nsecs

            camera_timestamp_epoch_secs = self.camera_image.header.stamp.secs
            camera_timestamp_epoch_nsecs = self.camera_image.header.stamp.nsecs

            filepath = DIR_PATH + "/data/simulator/"
            filename = str(light_timestamp_epoch_secs) + "_" + str(light_timestamp_epoch_nsecs) + "_" + str(
                camera_timestamp_epoch_secs) + "_" + str(camera_timestamp_epoch_nsecs) + "_" + str(
                state) + "_" + "%.3f" % self.next_traffic_light_dist + ".bmp"
            cv2.imwrite(filepath + filename, cv_image)
            rospy.loginfo("TL_DETECTOR: Wrote image: %s%s", filepath, filename)

        return state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closest to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if self.pose and (self.has_image or self.use_ground_truth):
            # find the closest traffic light coming up in front of us
            tl_idx = self.get_closest_waypoint(self.pose.pose, use_lights=True)
            if tl_idx >= 0:
                closest_light = self.lights[tl_idx]
                if closest_light:
                    # make sure car is within a reasonable distance to traffic light to reduce computation checks
                    next_traffic_light_dist = distance(closest_light.pose.pose.position, self.pose.pose.position)
                    rospy.logdebug("TL_DETECTOR: Distance to next traffic light: %.2f", next_traffic_light_dist)
                    if next_traffic_light_dist <= self.check_light_distance:
                        # find the stop line in front of the next traffic light
                        stop_line = self.config['stop_line_positions'][tl_idx]
                        stop_line_pose = Pose()
                        stop_line_pose.position.x = stop_line[0]
                        stop_line_pose.position.y = stop_line[1]
                        sl_wp_idx = self.get_closest_waypoint(stop_line_pose, use_lights=False)
                        if sl_wp_idx >= 0:
                            rospy.logdebug("TL_DETECTOR: next traffic light is # %s", tl_idx)
                            rospy.logdebug("TL_DETECTOR: next stop line is at waypoint %s", sl_wp_idx)
                            state = self.get_light_state(closest_light)
                            return sl_wp_idx, state
                        else:
                            rospy.logerr("TL_DETECTOR: didn't find stop line")
                    else:
                        rospy.logdebug("TL_DETECTOR: no traffic lights within %.1f units", self.check_light_distance)
            else:
                rospy.logerr("TL_DETECTOR: didn't find traffic light")
        else:
            rospy.logwarn("TL_DETECTOR: no pose found or we don't want to find lights")

        # Gavin TODO: Figure out - do we need this? When will light not return (or should not return?)
        # ANAND: though the code should never get here, this seems destructive
        # self.waypoints = None
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
