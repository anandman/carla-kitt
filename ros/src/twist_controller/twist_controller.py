
GAS_DENSITY = 2.858
ONE_MPH = 0.44704

from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
        accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):

        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        kp = 0.3
        ki = 0.1
        kd = 0.0
        mn = 0.0
        mx = 50.2 #TODO: ajaffer - what max value should be used?

        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        tau = 0.5
        ts = .02
        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()

    def control(self, current_vel, current_ang_vel, dbw_enabled, linear_vel, angular_vel):
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.

        current_vel = self.vel_lpf.filt(current_vel)

        # rospy.logwarn("Target velocity: {0}".format(linear_vel))
        # rospy.logwarn("Target Angular vel: {0}".format(angular_vel))
        # rospy.logwarn("Current velocity: {0}".format(current_vel))
        # rospy.logwarn("Current Angular velocity: {0}".format(current_ang_vel))
        # rospy.logwarn("Filtered velocity: {0}".format(self.vel_lpf.get()))

        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        if (abs(current_ang_vel - angular_vel) < 1e-7):
            rospy.logerr("ignoring steering")
            steering = 0
        else:
            dampned = steering * 0.90
            # TODO: ajaffer - try PID instead
            # rospy.logwarn("steering: {0} dampned: {1}".format(steering,
            #                                               dampned))
            steering = dampned

        vel_error = linear_vel - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0
        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            brake = 400 #N*m
        elif throttle < .1 and vel_error < 0:
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius # Torque N*m
            #TODO: ajaffer - use fuel_capacity to figure out brake

        # rospy.logerr("Vel_error: {0} Throttle: {1} Brake: {2} Steering: {"
        #              "3}".format(vel_error, throttle, brake, steering))

        return throttle, brake, steering
