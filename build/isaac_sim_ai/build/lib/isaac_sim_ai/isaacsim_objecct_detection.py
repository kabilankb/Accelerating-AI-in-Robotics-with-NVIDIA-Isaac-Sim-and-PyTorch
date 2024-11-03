import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
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
            '/jetson_webcam',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(Image, 'object_detection/image', 10)
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

        # Convert OpenCV image back to ROS Image message
        try:
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.publisher_.publish(ros_image)
        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

