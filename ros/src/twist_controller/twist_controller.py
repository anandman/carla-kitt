
GAS_DENSITY = 2.858
ONE_MPH = 0.44704

from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
        accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):

        min_speed = rospy.get_param("/min_speed", 0.1);
        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        kp = rospy.get_param("/kp", 0.3)
        ki = rospy.get_param("/ki", 0.1)
        kd = rospy.get_param("/kd", 0.0)
        mn = rospy.get_param("/min_throttle", 0.0)
        mx = rospy.get_param("/max_throttle", 0.2)
        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        kp_brake = rospy.get_param("/kp_brake", 0.9)
        ki_brake = rospy.get_param("/ki_brake", 0.001)
        kd_brake = rospy.get_param("/kd_brake", 0.6)
        mn_brake = rospy.get_param("/min_throttle", 0.0)
        mx_brake = abs(decel_limit)

        self.brake_controller = PID(kp_brake, ki_brake, kd_brake, mn_brake,
                                    mx_brake)

        tau = rospy.get_param("/lpf_tau", 0.5)
        ts = rospy.get_param("/lpf_ts", 0.02)
        self.vel_lpf = LowPassFilter(tau, ts)
        self.steer_lfp = LowPassFilter(tau, ts)
        # self.steer_lfp = LowPassFilter(0.2, 0.1)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.fuel_mass = self.fuel_capacity * GAS_DENSITY
        self.total_vehicle_mass = self.vehicle_mass + self.fuel_mass

        self.last_time = rospy.get_time()

    def control(self, current_vel, current_ang_vel, dbw_enabled, desired_linear_vel, desired_angular_vel):
        if not dbw_enabled:
            self.throttle_controller.reset()
            self.brake_controller.reset()
            return 0., 0., 0.

        #Low Pass Filter, filters out the high-frequency noise in velocity values
        current_vel = self.vel_lpf.filt(current_vel)

        steering = self.yaw_controller.get_steering(desired_linear_vel, desired_angular_vel, current_vel)
        #Low Pass Filter, filters out the high-frequency noise in steering values
        steering = self.steer_lfp.filt(steering)

        vel_error = desired_linear_vel - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        brake = 0.0
        throttle = 0.0

        if desired_linear_vel <= rospy.get_param(
                "/complete_stop_speed", 0.0):
            brake = rospy.get_param("/torque_complete_stop", 600)
            throttle = 0.0
            self.brake_controller.reset()
            rospy.logdebug("Complete stop [curr,desired,brake] %s,%s,%s",
                          current_vel,
                          desired_linear_vel,
                          brake)
        elif (vel_error < 0.0):
            decel = max(vel_error, self.decel_limit)
            decel = self.brake_controller.step(abs(decel), sample_time)
            brake = decel * self.total_vehicle_mass * self.wheel_radius #Torque
            throttle = 0.0
            self.throttle_controller.reset()
            rospy.logdebug("Slowing down [curr,desired,brake] %s,%s,%s",
                          current_vel,
                          desired_linear_vel,
                          brake)
        else:
            brake = 0.0
            throttle = self.throttle_controller.step(vel_error, sample_time)
            self.brake_controller.reset()
            rospy.logdebug("Speed up [curr,desired,throttle] %s,%s,%s",
                          current_vel,
                          desired_linear_vel,
                          throttle)

        return throttle, brake, steering
