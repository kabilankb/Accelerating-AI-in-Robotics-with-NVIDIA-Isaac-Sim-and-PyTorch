import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image as PILImage

# Create a color map for the segmentation classes
def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap

# Apply the colormap to the segmentation mask
def label_to_color_image(label):
    colormap = create_pascal_label_colormap()
    return colormap[label]

class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')
        self.subscription = self.create_subscription(
            Image,
            '/jetson_webcam',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Image, '/segmented_image', 10)
        self.bridge = CvBridge()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', weights='DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1')
        self.model.eval()
        self.preprocess = T.Compose([
            T.Resize(513),
            T.CenterCrop(513),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def listener_callback(self, msg):
        try:
            # Print the encoding of the incoming image
            self.get_logger().info(f"Image encoding: {msg.encoding}")
            
            # Try to convert the image based on its encoding
            if msg.encoding == '8UC3':
                # Directly convert the image data to a numpy array
                cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            else:
                # Handle other encodings using cv_bridge
                cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                
        except CvBridgeError as e:
            self.get_logger().error(f'Error converting image: {e}')
            return

        # Convert the OpenCV image to PIL format
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        # Preprocess the image for the model
        input_tensor = self.preprocess(pil_image)
        input_tensor = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        # Get the segmentation result
        output_predictions = output.argmax(0).byte().cpu().numpy()
        
        # Apply the colormap to the segmentation result
        colored_output = label_to_color_image(output_predictions).astype(np.uint8)
        
        # Convert the colored segmentation result back to ROS Image message format
        segmented_image = self.bridge.cv2_to_imgmsg(colored_output, encoding='rgb8')
        self.publisher.publish(segmented_image)

def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

