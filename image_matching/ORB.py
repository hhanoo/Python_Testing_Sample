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

    # Load base image (Initial Image)
    rgb_data, depth_data = sensor.get_data()
    base_img = rgb_data

    while True:
        # Retrieve sensor data
        rgb_data, depth_data = sensor.get_data()
        if rgb_data is None or depth_data is None:
            continue

        # ORB detection and matching
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(base_img, None)
        keypoints2, descriptors2 = orb.detectAndCompute(rgb_data, None)

        # Match keypoints using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        # Extract matched keypoint coordinates
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Calculate inlier ratio
        inlier_ratio = np.sum(mask) / len(mask)
        print(f"Inlier ratio: {inlier_ratio * 100:3.2f} %")

        # Visualize target image boundary
        h, w, _ = base_img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        current_img_with_box = cv2.polylines(rgb_data.copy(), [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        # Display result
        cv2.imshow("WINDOW", current_img_with_box.astype(np.uint8))

        # Handle user input
        msg = cv2.waitKey(1)
        if msg == ord("q") or msg & 0xFF == 27 or cv2.getWindowProperty('WINDOW', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Stop the sensor and close windows
    sensor.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
