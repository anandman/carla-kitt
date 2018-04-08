# Subscribers

tl_detector contains images that takes images from:

## /image_color =>

std_msgs/Header header
uint32 height
uint32 width
string encoding
uint8 is_bigendian
uint32 step
uint8[] data

More info @ http://docs.ros.org/api/sensor_msgs/html/msg/Image.html

## /current_pose =>
A Pose with reference coordinate frame and timestamp
Header header
Pose pose

(where Pose is:

A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation
)

## /base_waypoints =>

Header header
Waypoint[] waypoints

(Where waypoint is:
geometry_msgs/PoseStamped pose
geometry_msgs/TwistStamped twist

and Twist is:
This expresses velocity in free space broken into its linear and angular parts.
Vector3  linear
Vector3  angular
)

In other words, the waypoints are a list of (linear_v, angular_v) vectors

## WHEN IN SIMULATION: /vehicle/traffic_lights will provide the state and location of traffic lights IN 3D SPACE.
Thought -> Should we then extract images based on when the car is close to these states and get labelled data?
in 3D space -> this allows us to pin point the traffic light within an image (probably) so we can generate training data.
How else is this supposed to be used?

# PUBLISHES
/traffic_# waypoint =>
publishes the index of the base waypoint that is closest to the red light stop line.

# traffic_light_config
Contains the (x, y) coordinates of all traffic lights
- use get_closest_waypoint() of car to find closest traffic light stop line

