import cv2
import numpy as np
from rs_sensor import RSSensor  # Import the RSSensor class


def get_camera_parameters(sensor):
    """Retrieves the internal camera parameters from the sensor."""
    ppx, ppy, fx, fy = sensor.intr_params.ppx, sensor.intr_params.ppy, sensor.intr_params.fx, sensor.intr_params.fy
    camera_matrix = np.array([[fx, 0, ppx],
                              [0, fy, ppy],
                              [0, 0, 1]])
    camera_params = (camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2])
    return camera_matrix, camera_params


def main():
    # Initialize the sensor
    sensor = RSSensor()
    sensor.get_device_sn()
    sensor.start()

    # Retrieve camera parameters
    camera_matrix, camera_params = get_camera_parameters(sensor)

    while True:
        # Retrieve sensor data
        rgb_data, depth_data = sensor.get_data()
        if rgb_data is None or depth_data is None:
            continue

        gray = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2GRAY)

        # Detect Harris corner
        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(gray)

        # Draw circle at corner coordinate
        rgb_data = cv2.drawKeypoints(rgb_data, keypoints, None, (0, 0, 255),
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # Display result
        cv2.imshow("WINDOW", rgb_data.astype(np.uint8))

        # Handle user input
        msg = cv2.waitKey(1)
        if msg == ord("q") or msg & 0xFF == 27 or cv2.getWindowProperty('WINDOW', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Stop the sensor and close windows
    sensor.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()