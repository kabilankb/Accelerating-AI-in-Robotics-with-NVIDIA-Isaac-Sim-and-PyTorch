# Pose Estimation Node for ROS2

This ROS2 package performs human pose estimation using a Keypoint R-CNN model. The node processes images from a stereo camera, detects human keypoints, visualizes them using markers, and controls a robot to follow the nearest human.

## Features
- Human pose estimation using Keypoint R-CNN.
- Visualization of bounding boxes, keypoints, and skeletons in RViz.
- Command velocity publishing for human-following behavior.

## Prerequisites
- ROS2 Humble or later
- Python 3.8+
- Required Python libraries:
  ```bash
  pip install torch torchvision opencv-python numpy
  ```
- ROS2 Python packages:
  ```bash
  sudo apt install ros-humble-cv-bridge ros-humble-visualization-msgs ros-humble-sensor-msgs ros-humble-geometry-msgs
  ```

## Launch Instructions
1. Clone this repository into your ROS2 workspace:
   ```bash
   git clone <repository-url> ~/ros2_ws/src/pose_estimation_node
   cd ~/ros2_ws
   colcon build
   source install/setup.bash
   ```

2. Run the node:
   ```bash
   ros2 run pose_estimation_node pose_estimation_node
   ```

## Subscribed Topics
- `/front_stereo_camera/left_rgb/image_raw` (`sensor_msgs/Image`): Input image topic for pose estimation.

## Published Topics
- `output_image` (`sensor_msgs/Image`): Processed image with keypoints and bounding boxes.
- `visualization_marker_array` (`visualization_msgs/MarkerArray`): Markers for visualizing keypoints and skeletons.
- `cmd_vel` (`geometry_msgs/Twist`): Velocity commands for the robot to follow humans.

## Robot Behavior
- The robot identifies the closest human and adjusts its linear and angular velocities to follow.


