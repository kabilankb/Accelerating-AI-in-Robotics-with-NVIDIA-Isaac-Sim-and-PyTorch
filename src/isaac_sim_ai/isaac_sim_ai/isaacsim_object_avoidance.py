import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.subscription = self.create_subscription(
            Image,
            '/front_stereo_camera/left_rgb/image_raw',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(Image, 'object_detection/image', 10)
        self.cmd_vel_publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.bridge = CvBridge()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()
        self.coco_labels = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # For obstacle avoidance
        self.declare_parameter('forward_speed', 0.5)  # Forward speed
        self.declare_parameter('turn_speed', 0.5)     # Turn speed
        self.declare_parameter('stop_distance', 0.5)  # Distance threshold for stopping (in meters)
        self.declare_parameter('turn_distance', 1.0)  # Distance threshold for turning (in meters)
        self.forward_speed = self.get_parameter('forward_speed').get_parameter_value().double_value
        self.turn_speed = self.get_parameter('turn_speed').get_parameter_value().double_value
        self.stop_distance = self.get_parameter('stop_distance').get_parameter_value().double_value
        self.turn_distance = self.get_parameter('turn_distance').get_parameter_value().double_value

    def listener_callback(self, msg):
        self.get_logger().info(f"Received image with encoding: {msg.encoding}")
        try:
            if msg.encoding == '8UC3':
                # Convert ROS Image message to OpenCV image manually
                cv_image = self.bridge.imgmsg_to_cv2(msg)
                # Convert 8UC3 to BGR
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            else:
                # Convert ROS Image message to OpenCV image
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # Convert OpenCV image to PyTorch tensor
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        image_tensor = transform(cv_image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        # Perform object detection
        with torch.no_grad():
            detections = self.model(image_tensor)[0]

        # Draw transparent bounding boxes and add object names
        obstacle_detected = False
        closest_distance = float('inf')  # Initialize with infinity

        for i in range(len(detections['boxes'])):
            box = detections['boxes'][i].cpu().numpy().astype(int)
            score = detections['scores'][i].cpu().numpy()
            label = detections['labels'][i].cpu().numpy()
            label_name = self.coco_labels[label]

            if score > 0.5:  # Only draw boxes with a score above 0.5
                overlay = cv_image.copy()
                cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), -1)  # Filled green rectangle

                alpha = 0.4  # Transparency factor
                cv_image = cv2.addWeighted(overlay, alpha, cv_image, 1 - alpha, 0)

                # Put label text
                label_size, base_line = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                top = max(box[1], label_size[1])
                cv2.rectangle(cv_image, (box[0], top - label_size[1]), (box[0] + label_size[0], top + base_line), (0, 255, 0), cv2.FILLED)
                cv2.putText(cv_image, label_name, (box[0], top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Check obstacle distance and adjust behavior
                if label_name in ['car', 'truck', 'bus', 'motorcycle']:
                    distance = self.estimate_obstacle_distance(box)
                    if distance < closest_distance:
                        closest_distance = distance

        # Control logic based on closest detected obstacle distance
        if closest_distance < self.stop_distance:
            self.get_logger().warn(f"Obstacle too close: Stopping. Distance: {closest_distance}")
            self.stop_robot()
        elif closest_distance < self.turn_distance:
            self.get_logger().warn(f"Obstacle within turn distance: Turning. Distance: {closest_distance}")
            self.turn_robot()
        else:
            self.get_logger().info(f"No close obstacle: Moving forward. Distance: {closest_distance}")
            self.move_forward()

        # Convert OpenCV image back to ROS Image message
        try:
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.publisher_.publish(ros_image)
        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert image: {e}')

    def estimate_obstacle_distance(self, box):
        # Simple heuristic: use the height of the bounding box as a proxy for distance
        box_height = box[3] - box[1]
        focal_length = 525.0  # Example value, adjust based on your camera
        known_height = 1.75  # Example known height of the object in meters (e.g., average car height)
        distance = (known_height * focal_length) / box_height
        return distance

    def move_forward(self):
        twist = Twist()
        twist.linear.x = self.forward_speed  # Move forward
        twist.angular.z = 0.0  # No rotation
        self.cmd_vel_publisher_.publish(twist)

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0  # Stop moving forward
        twist.angular.z = 0.0  # No rotation
        self.cmd_vel_publisher_.publish(twist)

    def turn_robot(self):
        twist = Twist()
        twist.linear.x = 0.0  # Stop moving forward
        twist.angular.z = self.turn_speed  # Turn to avoid obstacle
        self.cmd_vel_publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

