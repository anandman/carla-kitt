
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

        #ajaffer: Tried with Highway track only, for parking lot we might
        # need to go to 0.2
        self.max_throttle = 1.0

        kp = 0.3
        ki = 0.1
        kd = 0.0
        mn = 0.0
        mx = self.max_throttle

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

        self.fuel_mass = self.fuel_capacity * GAS_DENSITY
        self.total_vehicle_mass = self.vehicle_mass + self.fuel_mass

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
            # rospy.loginfo("no need for steering")
            steering = 0

        vel_error = linear_vel - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0
        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            brake = 400
        elif throttle < (self.max_throttle/2.) and vel_error < 0:
            decel = max(vel_error, self.decel_limit)

            if (abs(decel) > self.brake_deadband):
                brake = abs(decel)*self.total_vehicle_mass*self.wheel_radius

        # rospy.loginfo("{0},{1},{2},{3},{4}".format(current_vel / ONE_MPH,
        #                                           vel_error, throttle, brake, steering))

        return throttle, brake, steering
