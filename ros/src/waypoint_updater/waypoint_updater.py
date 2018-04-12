#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

from scipy.interpolate import interp1d

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
TARGET_ACCEL = 5
MAX_JERK = 10
DECEL_DIST = 20
ACCEL_DIST = 30

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb, queue_size=1)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.current_waypoint_pub = rospy.Publisher('current_waypoint_id', Int32, queue_size=1,
                                                    latch=True)

        self.waypoints = []
        self.waypoint_speeds = []
        self.current_pose = None
        self.current_waypoint_id = 0
        self.traffic_wp = -1
        self.obstacle_wp = -1

        rospy.spin()

    def fake_stop(self, event=None):
        rospy.loginfo("fakestop")
        msg = Int32(400)
        self.traffic_cb(msg)
        return False

    def fake_start(self, event=None):
        rospy.loginfo("fakestart")
        msg = Int32(-1)
        self.traffic_cb(msg)
        return False

    def pose_cb(self, msg):
        """
        ROS callback for the car's current pose in the /world coordinate frame. Updated at 50Hz.
        We update the current waypoint id and publish the following LOOKAHEAD_WPS waypoints as
        well as the current_waypoint_id.

        :param msg: the current pose of the car in /world coordinates
        :type  msg: geometry_msgs.msg.PoseStamped
        """
        self.current_pose = msg
        if self.waypoints:
            self.update_current_waypoint(self.current_pose)
            self.publish_next_waypoints(self.current_waypoint_id, msg.header.stamp)
            speed = self.get_waypoint_velocity(self.current_waypoint_id)
            rospy.loginfo("At waypoint %s, wp speed %s", self.current_waypoint_id, speed)

    def update_current_waypoint(self, current_pose):
        """
        Find the next closest waypoint and update self.current_waypoint_id. If the current
        waypoint changes, publish the new one to the /current_waypoint_id ROS topic.

        :param current_pose: the latest pose of the car in /world coordinates
        :type  current_pose: geometry_msgs.msg.PoseStamped
        """
        new_waypoint_id = self.find_next_closest_waypoint(current_pose)
        if new_waypoint_id != self.current_waypoint_id:
            msg = Int32(data=new_waypoint_id)
            self.current_waypoint_pub.publish(msg)
        self.current_waypoint_id = new_waypoint_id

    def waypoints_cb(self, waypoints):
        """
        ROS callback for a waypoint list which is fired once at startup from a latched topic.

        :param waypoints: the list of waypoints in the /world frame
        :type  waypoints: styx_msgs.msg.Lane
        """
        rospy.loginfo("waypoints %s", len(waypoints.waypoints))
        self.waypoints = waypoints.waypoints
        self.waypoint_speeds = [wp.twist.twist.linear.x for wp in self.waypoints]
        print self.waypoints[0]

        rospy.Timer(rospy.Duration(1), self.fake_stop, oneshot=True)

    def traffic_cb(self, msg):
        """
        ROS callback for detected traffic lights. The value is -1 when no traffic light is detected
        or the waypoint ID of a detected traffic light in the red or yellow states. This will update
        the waypoint velocities to smoothly come to a stop or start.

        :param msg: traffic light waypoint id
        :type  msg: std_msgs.msg.Int32
        """
        if self.traffic_wp == msg.data:
            # rospy.loginfo("SKIP: traffic wp: %s", msg.data)
            return

        rospy.loginfo("CHANGE: traffic wp: %s", msg.data)
        self.traffic_wp = msg.data
        if msg.data >= 0:
            # Come to a stop by WP ID
            if self.traffic_wp <= self.current_waypoint_id:
                rospy.logwarn("traffic_wp %s is behind current wp: %s", self.traffic_wp,
                              self.current_waypoint_id)
                return

            start_accel_wp_id = self.wp_id_at_dist_before(self.traffic_wp, DECEL_DIST)
            # If we're past the desired start_accel_wp_id, start decelerating at our current location
            start_accel_wp_id = max(self.current_waypoint_id, start_accel_wp_id)
            stop_accel_wp_id = self.traffic_wp
            curr_speed = self.get_waypoint_velocity(start_accel_wp_id)
            target_speed = 0.0

            # TODO: use the TARGET_ACCEL rate to determine the stopping distance?
        else:
            # Removed blocking WP ID so slowly speed up
            start_accel_wp_id = self.current_waypoint_id
            stop_accel_wp_id = self.wp_id_at_dist_after(self.current_waypoint_id, ACCEL_DIST)
            curr_speed = self.get_waypoint_velocity(self.current_waypoint_id)
            target_speed = self.waypoints[stop_accel_wp_id].twist.twist.linear.x

            # TODO: use the TARGET_ACCEL rate to determine the stopping distance?

        actual_accel_dist = self.distance(self.waypoints, start_accel_wp_id, stop_accel_wp_id)
        rospy.loginfo("start_wp_id: %s, stop_wp_id: %s, stop_dist: %s", start_accel_wp_id,
                      stop_accel_wp_id, actual_accel_dist)

        # create a spline between that curr_speed and target_speed so we smoothly change acceleration
        x = [-1, 0, actual_accel_dist, actual_accel_dist + 1]
        y = [curr_speed, curr_speed, target_speed, target_speed]
        print(x)
        print(y)
        F = interp1d(x, y, kind='cubic')
        x = []
        y = []
        # Step through each waypoint and set the speed using the spline
        for wp_id in range(start_accel_wp_id, stop_accel_wp_id):
            dist = self.distance(self.waypoints, start_accel_wp_id, wp_id)
            speed = float(F(dist))
            x.append(dist)
            y.append(speed)
            self.set_waypoint_velocity(wp_id, speed)
        # Traffic wp id should be 0.
        self.set_waypoint_velocity(self.traffic_wp, 0.0)
        print x
        print y
        """
>>> from scipy.interpolate import interp1d
>>> F = interp1d([0, 1, 8, 9], [0, 0, 1, 1], kind='cubic')
>>> F(0)
array(-2.0816681711721685e-17)
>>> F(4)
array(0.4047619047619046)
>>> F(5)
array(0.5952380952380952)
>>> F(1)
array(-5.204170427930421e-17)
>>> F(2)
array(0.08333333333333323)
>>> import matplotlib.pyplot as plt
>>> x = range(0,10)
>>> plt.plot(x, F(x), label='cubic')
[<matplotlib.lines.Line2D object at 0x10678ef90>]
>>> plt.show()
        """

    def obstacle_cb(self, msg):
        """
        Note: currently ignored!

        ROS callback for detected obstacles. The value is -1 when no obstacle is detected
        or the waypoint ID of a detected obstacle. This will update the waypoint velocities to
        smoothly com to a stop or start.

        :param msg: obstacle waypoint id
        :type  msg: std_msgs.msg.Int32
        """
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        self.obstacle_wp = msg.data
        rospy.loginfo("obstacle wp: %s", msg)

    def get_waypoint_velocity(self, waypoint_id):
        # return self.waypoints[waypoint_id].twist.twist.linear.x
        return self.waypoint_speeds[waypoint_id]

    def set_waypoint_velocity(self, waypoint_id, velocity):
        # waypoints[waypoint].twist.twist.linear.x = velocity
        self.waypoint_speeds[waypoint_id] = velocity

    def find_next_closest_waypoint(self, pose):
        assert self.waypoints, "Waypoint list is blank"
        closest = 9999999
        closest_id = 0
        # Start looking 10 waypoints back and iterate forward until we hit a local minima
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

    def publish_next_waypoints(self, start_id, stamp):
        waypoints = self.waypoints[start_id + 1 : start_id + 1 + LOOKAHEAD_WPS]
        msg = Lane()
        msg.header.frame_id = '/world'
        msg.header.stamp = stamp
        for i in xrange(len(waypoints)):
            wp_msg = Waypoint()
            wp_msg.pose = waypoints[i].pose
            wp_msg.twist.twist.linear.x = waypoints[i].twist.twist.linear.x
            msg.waypoints.append(wp_msg)

        # rospy.loginfo("Published final_waypoints starting at %s", start_id)
        self.final_waypoints_pub.publish(msg)

    def wp_id_at_dist_before(self, start_wp_id, target_dist):
        """
        Step backwards in waypoint list until the distance reaches target_dist

        :param target_dist: the distance in meters
        :returns: the waypoint id within the target_dist
        """
        stop_wp_id = start_wp_id
        dist = 0
        last_pos = self.waypoints[stop_wp_id].pose.pose.position
        while dist < target_dist and stop_wp_id >= 0:
            stop_wp_id -= 1
            next_pos = self.waypoints[stop_wp_id].pose.pose.position
            dist += dl(last_pos, next_pos)
            last_pos = next_pos

        return stop_wp_id

    def wp_id_at_dist_after(self, start_wp_id, target_dist):
        """
        Step forwards in waypoint list until the distance reaches target_dist

        :param target_dist: the distance in meters
        :returns: the waypoint id within the target_dist
        """
        stop_wp_id = start_wp_id
        dist = 0
        last_pos = self.waypoints[stop_wp_id].pose.pose.position
        while dist < target_dist and stop_wp_id >= 0:
            stop_wp_id += 1
            next_pos = self.waypoints[stop_wp_id].pose.pose.position
            dist += dl(last_pos, next_pos)
            last_pos = next_pos

        return stop_wp_id

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
