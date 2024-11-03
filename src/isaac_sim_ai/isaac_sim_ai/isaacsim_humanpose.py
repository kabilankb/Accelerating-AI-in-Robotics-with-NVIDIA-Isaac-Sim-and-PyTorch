import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__('pose_estimation_node')
        self.get_logger().info('PoseEstimationNode initialized')
        self.subscription = self.create_subscription(Image, 'input_image', self.image_callback, 10)
        self.image_publisher = self.create_publisher(Image, 'output_image', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
        self.bridge = CvBridge()
        
        self.device = torch.device("cpu")
        
        weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=weights).to(self.device)
        self.model.eval()
        
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def image_callback(self, msg):
        self.get_logger().info('Received an image')
        try:
            if msg.encoding != 'bgr8':
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
                    self.get_logger().info('Converting image encoding from 8UC3 to BGR')
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                else:
                    self.get_logger().error(f'Unsupported image encoding: {msg.encoding}')
                    return
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        self.get_logger().info('Processing image for pose estimation')
        
        resized_image = cv_image
        input_tensor = self.transforms(resized_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        self.get_logger().info(f'Model outputs: {outputs}')
        
        if len(outputs) > 0 and 'keypoints' in outputs[0]:
            keypoints = outputs[0]['keypoints'].cpu().numpy()
            scores = outputs[0]['scores'].cpu().numpy()
            high_score_idxs = np.where(scores > 0.5)[0]
            
            self.get_logger().info(f'High score indices: {high_score_idxs}')
            
            marker_array = MarkerArray()
            marker_id = 0
            
            for idx in high_score_idxs:
                kpts = keypoints[idx]
                for i, point in enumerate(kpts):
                    if point[2] > 0.5:  # Check confidence
                        cv2.circle(resized_image, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)
                        marker = Marker()
                        marker.header.frame_id = "camera_link"
                        marker.header.stamp = self.get_clock().now().to_msg()
                        marker.ns = "keypoints"
                        marker.id = marker_id
                        marker_id += 1
                        marker.type = Marker.SPHERE
                        marker.action = Marker.ADD
                        marker.pose.position.x = float(point[0]) / 100  # Adjust scaling if necessary
                        marker.pose.position.y = float(point[1]) / 100
                        marker.pose.position.z = 0.0
                        marker.pose.orientation.w = 1.0
                        marker.scale.x = 0.02
                        marker.scale.y = 0.02
                        marker.scale.z = 0.02
                        marker.color.a = 1.0
                        marker.color.r = 0.0
                        marker.color.g = 1.0
                        marker.color.b = 0.0
                        marker_array.markers.append(marker)
                    
                skeleton_pairs = [
                    (5, 6),  # Shoulders
                    (5, 7), (7, 9),  # Left arm
                    (6, 8), (8, 10),  # Right arm
                    (11, 12),  # Hips
                    (11, 13), (13, 15),  # Left leg
                    (12, 14), (14, 16)  # Right leg
                ]
                
                for pair in skeleton_pairs:
                    if (kpts[pair[0]][2] > 0.5) and (kpts[pair[1]][2] > 0.5):  # Check confidence for keypoints
                        cv2.line(resized_image, 
                                 (int(kpts[pair[0]][0]), int(kpts[pair[0]][1])),
                                 (int(kpts[pair[1]][0]), int(kpts[pair[1]][1])), 
                                 (0, 255, 0), 2)
                        marker = Marker()
                        marker.header.frame_id = "camera_link"
                        marker.header.stamp = self.get_clock().now().to_msg()
                        marker.ns = "skeleton"
                        marker.id = marker_id
                        marker_id += 1
                        marker.type = Marker.LINE_STRIP
                        marker.action = Marker.ADD
                        marker.scale.x = 0.01
                        marker.color.a = 1.0
                        marker.color.r = 0.0
                        marker.color.g = 1.0
                        marker.color.b = 0.0
                        
                        point1 = Marker().pose.position
                        point1.x = float(kpts[pair[0]][0]) / 100
                        point1.y = float(kpts[pair[0]][1]) / 100
                        point1.z = 0.0
                        point2 = Marker().pose.position
                        point2.x = float(kpts[pair[1]][0]) / 100
                        point2.y = float(kpts[pair[1]][1]) / 100
                        point2.z = 0.0
                        
                        marker.points = [point1, point2]
                        
                        marker_array.markers.append(marker)
            
            self.marker_publisher.publish(marker_array)
            
            output_msg = self.bridge.cv2_to_imgmsg(resized_image, 'bgr8')
            self.image_publisher.publish(output_msg)
            self.get_logger().info('Published pose estimation result')
        else:
            self.get_logger().warning('No keypoints detected or low confidence scores')

def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
