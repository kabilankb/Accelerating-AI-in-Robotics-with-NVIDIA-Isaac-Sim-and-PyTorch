import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
from transformers import pipeline

class CubeFollower(Node):
    def __init__(self):
        super().__init__('cube_follower')
        self.subscription_image = self.create_subscription(Image, '/rgb', self.image_callback, 10)
        self.subscription_command = self.create_subscription(String, 'nlp_command', self.command_callback, 10)
        self.publisher_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        
        self.bridge = CvBridge()

        # Use GPU if available, otherwise fallback to CPU
        self.nlp = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad', device=0)
        self.current_target_color = None

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        if self.current_target_color:
            mask = self.detect_color(cv_image, self.current_target_color)
            if mask is not None:
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    self.follow_cube(x, w, cv_image.shape[1])

        cv2.imshow('Color Detection', cv_image)
        cv2.waitKey(1)

    def detect_color(self, image, color):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if color == 'green':
            lower_bound = (40, 50, 50)
            upper_bound = (80, 255, 255)
        elif color == 'yellow':
            lower_bound = (20, 100, 100)
            upper_bound = (30, 255, 255)
        elif color == 'red':
            lower_bound1 = (0, 100, 100)
            upper_bound1 = (10, 255, 255)
            lower_bound2 = (160, 100, 100)
            upper_bound2 = (180, 255, 255)
            mask1 = cv2.inRange(hsv_image, lower_bound1, upper_bound1)
            mask2 = cv2.inRange(hsv_image, lower_bound2, upper_bound2)
            return cv2.bitwise_or(mask1, mask2)

        return cv2.inRange(hsv_image, lower_bound, upper_bound)

    def follow_cube(self, x, w, image_width):
        twist = Twist()
        center_x = x + w // 2
        if center_x < image_width // 3:
            twist.angular.z = 0.2  # Turn left
        elif center_x > 2 * image_width // 3:
            twist.angular.z = -0.2  # Turn right
        else:
            twist.linear.x = 0.5  # Move forward
        self.publisher_cmd_vel.publish(twist)

    def command_callback(self, msg):
        command = msg.data.lower()
        if 'green' in command:
            self.current_target_color = 'green'
        elif 'yellow' in command:
            self.current_target_color = 'yellow'
        elif 'red' in command:
            self.current_target_color = 'red'

        self.get_logger().info(f'Target color set to {self.current_target_color}')

def main(args=None):
    # Suppress TensorFlow warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    rclpy.init(args=args)
    cube_follower = CubeFollower()
    rclpy.spin(cube_follower)
    cube_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

