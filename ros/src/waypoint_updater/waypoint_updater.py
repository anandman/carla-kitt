#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.waypoints = None
        self.pose = None
        self.current_waypoint_id = 0
        self.waypoints_2d = None
        self.waypoint_tree = None

        # rospy.spin()
        self.loop()

    def loop(self):
        rospy.logwarn("starting loop")
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            # rospy.logwarn("self.pose and self.waypoints and self.waypoint_tree %s/%s/%s", self.pose is not None, self.waypoints is not None, self.waypoint_tree is not None)
            if self.pose and self.waypoints and self.waypoint_tree:
                # rospy.logwarn("finding closes waypoint")
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                # rospy.logwarn("closest_waypoint_idx :%s ", closest_waypoint_idx)
                self.publish_next_waypoints(closest_waypoint_idx)
            else:
                if (self.pose is None):
                    rospy.logwarn('pose not ready')
                if (self.waypoints is None):
                    rospy.logwarn('waypoints not ready')
                if (self.waypoint_tree is None):
                    rospy.logwarn('waypoint_tree not ready')
            rate.sleep()

    def get_closest_waypoint_idx(self):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest coord
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def pose_cb(self, msg):
        """
        ROS callback for the car's current pose in the /world coordinate frame. Updated at 50Hz.
        We update the
        """
        self.pose = msg
        # if self.waypoints:
        #     self.current_waypoint_id = self.find_next_closest_waypoint(self.current_pose)
        #     self.publish_next_waypoints(self.current_waypoint_id, msg.header.stamp)

    def waypoints_cb(self, waypoints):
        """
        ROS callback for a waypoint list which is fired once at startup from a latched topic.

        :param waypoints: the list of waypoints in the /world frame
        :type  waypoints: styx_msgs.msg.Lane
        """
        rospy.loginfo("waypoints %s", len(waypoints.waypoints))
        self.waypoints = waypoints.waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
            rospy.logwarn("waypoints_2d %s ", len(self.waypoints_2d))

    def traffic_cb(self, msg):
        """
        ROS callback for detected traffic lights. This returns -1 when the camera is turned on in
        the simulator.

        TODO: fill out more details once I understand it more.

        :param msg: traffic light waypoint id
        :type  msg: std_msgs.msg.Int32
        """
        # TODO: Callback for /traffic_waypoint message. Implement
        # rospy.loginfo("traffic wp: %s", msg)

    def obstacle_cb(self, msg):
        """
        ROS callback for detected obstacles.

        TODO: fill out more details once I understand it more.

        :param msg: obstacle waypoint id
        :type  msg: std_msgs.msg.Int32
        """
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        rospy.loginfo("obstacle wp: %s", msg)

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def find_next_closest_waypoint(self, pose):
        assert self.waypoints, "Waypoint list is blank"
        closest = 9999999
        closest_id = 0
        # I assume the car is only going forward in the waypoint list
        start_id = max(0, self.current_waypoint_id - 10)
        for i in xrange(start_id, len(self.waypoints)):
            dist = dl(pose.pose.position, self.waypoints[i].pose.pose.position)
            if dist < closest:
                closest_id = i
                closest = dist
            elif self.current_waypoint_id > 0:
                # This means we're getting further away from the closest waypoint. Except if we're
                # starting from the beginning we can't assume the local minima is the closest.
                break

        return closest_id

    def publish_next_waypoints(self, start_id):
        msg = Lane()
        msg.header.frame_id = '/world'
        # msg.header.stamp = stamp
        msg.waypoints = self.waypoints[start_id + 1 : start_id + 1 + LOOKAHEAD_WPS]
        rospy.logdebug("Published final_waypoints starting at %s", start_id)
        self.final_waypoints_pub.publish(msg)

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        # dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

def dl(pos1, pos2):
    return math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2)

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
