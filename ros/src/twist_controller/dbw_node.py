#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math
from std_msgs.msg import Int32

from twist_controller import Controller

'''
Subscribes to `/twist_cmd` message, which provides the desired linear and 
angular velocities, to `/current_velocity` message for the current linear and 
angular velocities.

Publishes the proposed throttle, brake, and steer values. 
'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        self.controller = Controller(vehicle_mass=vehicle_mass,
                                     fuel_capacity=fuel_capacity,
                                     brake_deadband=brake_deadband,
                                     decel_limit=decel_limit,
                                     accel_limit=accel_limit,
                                     wheel_radius=wheel_radius,
                                     wheel_base=wheel_base,
                                     steer_ratio=steer_ratio,
                                     max_lat_accel=max_lat_accel,
                                     max_steer_angle=max_steer_angle)

        rospy.loginfo('Staring DBW Node')
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cb)
        rospy.Subscriber('/current_velocity', TwistStamped,
                         self.current_velocity_cb)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)
        # rospy.Subscriber('current_waypoint_id', Int32, self.waypoint_cb)

        #Current linear velocity
        self.current_vel = None

        #Current angular velocity
        self.current_ang_vel = None

        #False if the car is being driven manually, e.g., by a human
        self.dbw_enabled = None

        #Desired linear velocity
        self.desired_linear_vel = None

        #Desired angular velocity
        self.desired_angular_vel = None

        #Throttle, Steering and Brake values, which will be published
        self.throttle = self.steering = self.brake = 0

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)  # 50Hz
        while not rospy.is_shutdown():
            if not None in (self.current_vel,
                            self.desired_linear_vel,
                            self.desired_angular_vel):
                self.throttle, self.brake, self.steering = \
                    self.controller.control(
                        self.current_vel,
                        self.current_ang_vel,
                        self.dbw_enabled,
                        self.desired_linear_vel,
                        self.desired_angular_vel)

            if self.dbw_enabled:
                self.publish(self.throttle, self.brake, self.steering)
            rate.sleep()


    def twist_cb(self, msg):
        """Callback for the `/twist_cmd` topic.

        Receives the Twist Command messages containing the desired linear and
        angular velocities, which are published by pure_pursuit.

        """
        self.desired_linear_vel = msg.twist.linear.x
        self.desired_angular_vel = msg.twist.angular.z

    def dbw_enabled_cb(self, msg):
        self.dbw_enabled = msg

    def waypoint_cb(self, waypoint):
        rospy.loginfo("at waypoint: {0}".format(waypoint))


    def current_velocity_cb(self, msg):
        """Callback for the `/current_velocity` topic.

        Receives the Twist Command messages containing the current linear and
        angular velocities, which are published by styx_serve.

        """
        self.current_vel = msg.twist.linear.x
        self.current_ang_vel = msg.twist.angular.z


    def publish(self, throttle, brake, steer):
        """Publisher for the Steering, Throttle and Brake commands.

        It publishes the following topics, which are subscribed by the styx_server:

        /vehicle/steering_cmd
        /vehicle/throttle_cmd
        /vehicle/brake_cmd

        """
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
