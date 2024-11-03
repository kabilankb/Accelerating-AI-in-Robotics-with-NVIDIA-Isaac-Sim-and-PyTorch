import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class GreenBallTracker(Node):
    def __init__(self):
        super().__init__('green_ball_tracker')
        
        # Subscription to the camera topic
        self.subscription = self.create_subscription(
            Image,
            '/rgb',  # Change this to your camera topic
            self.listener_callback,
            10)
        
        # Publisher for velocity commands
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Publisher for ball coordinates
        self.coordinates_publisher = self.create_publisher(String, 'ball_coordinates', 10)
        
        # Publisher for processed image visualization
        self.processed_image_publisher = self.create_publisher(Image, 'processed_image', 10)
        
        # Bridge to convert ROS Image messages to OpenCV images
        self.br = CvBridge()

        # Parameters for control
        self.k_p = 0.002  # Proportional control gain
        self.linear_speed = 0.2
        self.max_angular_speed = 1.0  # Maximum angular speed

    def listener_callback(self, data):
        self.get_logger().info('Receiving video frame')

        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(data)

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

        # Define the range for the color green in HSV
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])

        # Create a mask to detect green color
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        twist = Twist()

        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the coordinates of the bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Calculate the center of the ball
            ball_center_x = x + w // 2
            ball_center_y = y + h // 2

            # Publish the coordinates
            coordinates = f'x: {ball_center_x}, y: {ball_center_y}'
            self.coordinates_publisher.publish(String(data=coordinates))

            # Draw the bounding box and center point on the image
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(current_frame, (ball_center_x, ball_center_y), 5, (0, 0, 255), -1)

            # Convert the image back to ROS message and publish it
            processed_image_msg = self.br.cv2_to_imgmsg(current_frame, encoding='bgr8')
            self.processed_image_publisher.publish(processed_image_msg)

            # Control logic to follow the ball
            frame_center_x = current_frame.shape[1] // 2
            error_x = frame_center_x - ball_center_x

            angular_speed = self.k_p * error_x
            if abs(angular_speed) > self.max_angular_speed:
                angular_speed = np.sign(angular_speed) * self.max_angular_speed

            # Adjust linear and angular speeds based on error
            if abs(error_x) < 50:  # Ball is roughly centered
                twist.linear.x = self.linear_speed
                twist.angular.z = 0.0
            else:
                twist.linear.x = 0.0
                twist.angular.z = angular_speed

            self.get_logger().info(f'Ball Coordinates: x={ball_center_x}, y={ball_center_y}, error_x={error_x}, angular_speed={angular_speed}')
        else:
            # Stop the robot if no ball is detected
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_vel_publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    green_ball_tracker = GreenBallTracker()
    rclpy.spin(green_ball_tracker)
    green_ball_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

