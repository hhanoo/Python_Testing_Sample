import pyrealsense2 as rs
import numpy as np


class RSSensor:
    def __init__(self):
        """Initialize RealSense sensor."""
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.intr_params = None

    def get_device_sn(self):
        """Retrieve and print the device serial number."""
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        serial_number = device.get_info(rs.camera_info.serial_number)
        print(f"Device Serial Number: {serial_number}")
        return serial_number

    def start(self):
        """Start the sensor and retrieve intrinsic parameters."""
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        profile = self.pipeline.start(self.config)
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intr_params = color_stream.get_intrinsics()

    def get_data(self):
        """Retrieve RGB and depth data from the sensor."""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        rgb_data = np.asanyarray(color_frame.get_data())
        depth_data = np.asanyarray(depth_frame.get_data())
        return rgb_data, depth_data

    def stop(self):
        """Stop the sensor."""
        self.pipeline.stop()
