import cv2
import numpy as np
from rs_sensor import RSSensor  # Import the RSSensor class


def get_camera_parameters(sensor):
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
        corners = cv2.cornerHarris(gray, 3, 3, 0.04)
        coord = np.where(corners >= 0.01 * corners.max())
        coord = np.stack([coord[1], coord[0]], axis=-1)

        # Draw circle at corner coordinate
        for x, y in coord:
            cv2.circle(rgb_data, (x, y), 5, (0, 0, 255), 1, cv2.LINE_AA)

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
