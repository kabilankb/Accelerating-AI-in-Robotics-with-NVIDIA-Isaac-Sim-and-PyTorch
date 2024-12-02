import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import math
import time

class SpotGaitPublisher(Node):
    def __init__(self):
        super().__init__('spot_gait_publisher')
        
        # Publisher for joint commands
        self.publisher_ = self.create_publisher(JointTrajectory, '/forward_position_controller/commands', 10)
        
        # Timer to publish walking commands at a regular interval
        self.timer = self.create_timer(0.2, self.publish_gait_command)  # Adjust timing for smoother gait
        
        # Joint names
        self.joint_names = [
            'front_right_leg_joint', 'front_right_thigh_joint', 'front_right_shin_joint',
            'front_left_leg_joint', 'front_left_thigh_joint', 'front_left_shin_joint',
            'rear_right_leg_joint', 'rear_right_thigh_joint', 'rear_right_shin_joint',
            'rear_left_leg_joint', 'rear_left_thigh_joint', 'rear_left_shin_joint'
        ]
        
        # Gait phase parameter
        self.gait_phase = 0

    def publish_gait_command(self):
        # Create a JointTrajectory message
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        # Calculate desired positions for each joint based on the gait phase
        point = JointTrajectoryPoint()
        
        # Define a simple walking cycle using a sine wave for joint positions
        point.positions = [
            # Right front leg cycle (example)
            0.2 * math.sin(self.gait_phase), -0.2 * math.cos(self.gait_phase), 0.1 * math.sin(self.gait_phase),
            # Left front leg cycle (offset by pi for alternating movement)
            0.2 * math.sin(self.gait_phase + math.pi), -0.2 * math.cos(self.gait_phase + math.pi), 0.1 * math.sin(self.gait_phase + math.pi),
            # Right rear leg cycle
            0.2 * math.sin(self.gait_phase + math.pi), -0.2 * math.cos(self.gait_phase + math.pi), 0.1 * math.sin(self.gait_phase + math.pi),
            # Left rear leg cycle
            0.2 * math.sin(self.gait_phase), -0.2 * math.cos(self.gait_phase), 0.1 * math.sin(self.gait_phase)
        ]
        
        # Set a velocity for the gait movement
        point.velocities = [0.5] * len(self.joint_names)  # Example velocity, adjust as needed
        point.time_from_start.sec = 1  # Timing for each phase, adjust for smoother movement

        # Add the point to the trajectory message
        msg.points.append(point)
        
        # Publish the message
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published walking gait command at phase {self.gait_phase}')
        
        # Increment gait phase for the next step in the cycle
        self.gait_phase += 0.1  # Adjust increment for gait speed

def main(args=None):
    rclpy.init(args=args)
    spot_gait_publisher = SpotGaitPublisher()
    
    try:
        rclpy.spin(spot_gait_publisher)
    except KeyboardInterrupt:
        pass
    
    spot_gait_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

