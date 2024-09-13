import time
import cv2
import numpy as np
import pyrealsense2 as rs
from realsense_depth import *
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class Cam3dNode(Node):
    def __init__(self):
        super().__init__('cam_3d_node')
        self.publisher_ = self.create_publisher(Image, 'camera/image', 10)
        self.bridge = CvBridge()

        self._model = YOLO('models/hands_with_human_last_obb.pt')
        self.shut_down = False
        self._show = True
        self._show_color = True
        self.frame = np.zeros((640, 480))
        
        # Initialize camera
        self.cam_open()
        
        # Start processing loop
        self.cam_thread = threading.Thread(target=self.camera)
        self.cam_thread.start()

    def cam_open(self):
        try:
            self.dc = DepthCamera()
        except Exception as exc:
            self.get_logger().error(f'Failed to open camera: {exc}')
            pass

    def camera(self):
        while rclpy.ok():
            ret, depth_frame, color_frame = self.dc.get_frame()
            if not ret:
                continue
            
            results = self._model.predict(color_frame, verbose=False)
            annotator = Annotator(color_frame.copy())
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]
                    annotator.box_label(b, self._model.names[int(box.cls)])
            
            annotated_img = annotator.result()
            self.publish_image(annotated_img)
            
            if self.shut_down:
                break
    
    def publish_image(self, frame):
        msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.publisher_.publish(msg)


if __name__ == '__main__':
    rclpy.init(args=args)
    node = Cam3dNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


