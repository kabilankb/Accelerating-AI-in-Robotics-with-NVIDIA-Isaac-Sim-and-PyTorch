import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from torchvision import models, transforms
import torch
import cv2
import numpy as np
from PIL import Image as PILImage
import matplotlib.pyplot as plt

class MonoDepthNode(Node):
    def __init__(self):
        super().__init__('mono_depth_node')
        self.subscription = self.create_subscription(
            Image,
            '/jetson_webcam',  # Change this to your image topic
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.publisher = self.create_publisher(Image, 'depth_map_topic', 10)
        self.bridge = CvBridge()

        # Load a pre-trained model (replace this with a depth estimation model as needed)
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.model.eval()

        # Define the image transform
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def listener_callback(self, msg):
        try:
            # Decoding: Handle '8UC3' encoding manually
            if msg.encoding == '8UC3':
                self.get_logger().warn(f"Received image with encoding: {msg.encoding}. Converting manually.")
                cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')  # Decoding with CvBridge
        except CvBridgeError as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        # Convert the image to PIL format
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        # Preprocess the image
        input_tensor = self.preprocess(pil_image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        # Perform inference
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]

        # Convert the output to a depth map
        depth_map = torch.sigmoid(output[0]).squeeze().numpy()

        # Normalize the depth map for visualization (0-1 range)
        normalized_depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        # Apply custom colormap for depth visualization
        depth_map_vis = self.apply_custom_colormap(normalized_depth_map)

        # Convert to 3-channel image (RGB)
        depth_map_vis_rgb = cv2.cvtColor(depth_map_vis, cv2.COLOR_RGBA2RGB)

        # Encoding: Convert the depth map to a ROS Image message
        depth_map_msg = self.bridge.cv2_to_imgmsg(depth_map_vis_rgb, encoding="rgb8")

        # Publish the depth map
        self.publisher.publish(depth_map_msg)

    def apply_custom_colormap(self, depth_map):
        colormap = plt.cm.plasma  # You can change the colormap here
        colored_depth_map = (colormap(depth_map) * 255).astype(np.uint8)
        return colored_depth_map

def main(args=None):
    rclpy.init(args=args)
    node = MonoDepthNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

