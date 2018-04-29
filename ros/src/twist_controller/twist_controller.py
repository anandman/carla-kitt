
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

        tau = rospy.get_param("/lpf_tau", 0.5)
        ts = rospy.get_param("/lpf_ts", 0.02)
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

    def control(self, current_vel, current_ang_vel, dbw_enabled, desired_linear_vel, desired_angular_vel):
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.

        #Low Pass Filter, filters out the high-frequency noise
        current_vel = self.vel_lpf.filt(current_vel)

        steering = self.yaw_controller.get_steering(desired_linear_vel, desired_angular_vel, current_vel)
        if (abs(current_ang_vel - desired_angular_vel) < 1e-7):
            steering = 0

        vel_error = desired_linear_vel - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0
        decel = max(vel_error, self.decel_limit)
        if desired_linear_vel <= rospy.get_param(
                "/desired_speed_limit_apply_full_brakes", 1.75) \
                and current_vel <= rospy.get_param(
                "/current_speed_limit_apply_full_brakes", 3.50):
            throttle = 0
            brake = max(abs(decel)*self.total_vehicle_mass*self.wheel_radius,
                        rospy.get_param("/torque_complete_stop", 400))

            rospy.loginfo("Applying brakes (Complete Stop) [curr_vel,des_vel,"
                          "err,"
                          "t,b,"
                          "s] [{0},"
                          "{1},"
                          "{2},"
                          "{3},"
                          "{4},"
                          "{5}]".format(current_vel,
                                        desired_linear_vel,
                                        vel_error,
                                        throttle,
                                        brake,
                                        steering))
        elif throttle < rospy.get_param("/throttle_limit_apply_brakes",
                                        0.1) and vel_error < 0:
            if (abs(decel) > self.brake_deadband):
                brake = abs(decel)*self.total_vehicle_mass*self.wheel_radius
                rospy.loginfo("Applying brakes [curr_vel,des_vel,err,t,b,"
                              "s] [{0},"
                              "{1},"
                              "{2},"
                              "{3},"
                              "{4},"
                              "{5}]".format(current_vel,
                                            desired_linear_vel,
                                            vel_error,
                                            throttle,
                                            brake,
                                            steering))
            else:
                rospy.loginfo("decel less than brake-deadband, ignoring "
                              "brakes: {0}".format(abs(decel)))

        return throttle, brake, steering
