import pyrealsense2 as rs
import numpy as np


class RSSensor:
    def __init__(self, sensor_index=0):
        """Initialize RealSense sensor with a specific serial number."""
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.intr_params = None
        self.serial_number = self._get_connected_devices()[sensor_index]
        self.config.enable_device(self.serial_number)

    @staticmethod
    def _get_connected_devices():
        """Retrieve all connected RealSense devices and return their serial numbers."""
        context = rs.context()
        devices = context.query_devices()
        serial_numbers = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
        return serial_numbers  # Return Serial Number list for all connected cameras

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
        print(f"Camera {self.serial_number} started.")

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
        print(f"Camera {self.serial_number} stopped.")

    def get_camera_parameters(self):
        """Get camera parameters."""
        if self.intr_params is None:
            raise RuntimeError("Intrinsic parameters are not available. Start the camera first.")

        ppx, ppy, fx, fy = self.intr_params.ppx, self.intr_params.ppy, self.intr_params.fx, self.intr_params.fy
        camera_matrix = np.array([[fx, 0, ppx],
                                  [0, fy, ppy],
                                  [0, 0, 1]])
        camera_params = (camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2])
        return camera_matrix, camera_params
