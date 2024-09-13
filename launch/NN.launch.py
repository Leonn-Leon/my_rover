from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_cam_rover',
            executable='cam_test',
            name='cam_3d_node',
            output='screen'
        )
    ])

