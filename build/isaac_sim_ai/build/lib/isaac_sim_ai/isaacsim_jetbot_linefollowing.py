import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
import numpy as np

class HolonomicLineFollower(Node):
    def __init__(self):
        super().__init__('kaya_line_follower')
        self.bridge = CvBridge()
        
        # Subscribe to the camera feed
        self.image_sub = self.create_subscription(
            Image,
            '/rgb',  # Update to your camera topic
            self.image_callback,
            10)
        
        # Publisher for holonomic velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def image_callback(self, msg):
        # Convert ROS image message to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Convert image to HSV color space to detect yellow
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define HSV range for yellow color (adjust as needed)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
        
        # Calculate moments to find the center of the yellow area
        moments = cv2.moments(mask)
        
        # If there is a yellow area detected in the frame
        if moments['m00'] > 0:
            # Calculate the centroid of the yellow line
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            
            # Calculate error based on the center of the frame
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2
            error_x = cx - frame_center_x
            error_y = cy - frame_center_y
            
            # Create a Twist message for holonomic control
            twist = Twist()
            
            # Forward movement proportional to the y-position error
            twist.linear.x = 0.2  # Constant speed forward
            
            # Sideways movement (holonomic control) to correct x error
            twist.linear.y = -float(error_x) / 200  # Adjust based on error_x
            
            # Rotation to align with line direction
            twist.angular.z = -float(error_x) / 300  # Adjust based on error_x
            
            self.cmd_vel_pub.publish(twist)
            self.get_logger().info(f"Yellow line detected at x={cx}, y={cy}, adjusting with error_x={error_x}, error_y={error_y}")
        
        else:
            # If no yellow line is detected, stop the robot
            twist = Twist()
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            self.get_logger().info("Yellow line not detected, stopping.")

def main(args=None):
    rclpy.init(args=args)
    line_follower = HolonomicLineFollower()
    rclpy.spin(line_follower)
    line_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

