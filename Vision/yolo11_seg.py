import torch
import torchvision

import random
import cv2
import numpy as np
from rs_sensor import RSSensor  # Import the RSSensor class
from ultralytics import YOLO

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())


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

    # Load model
    model = YOLO("../model/yolo11x-seg.pt")
    classes_ids = [list(model.names.values()).index(clas) for clas in list(model.names.values())]
    colors = [random.choices(range(256), k=3) for _ in classes_ids]

    while True:
        # Retrieve sensor data
        rgb_data, depth_data = sensor.get_data()
        if rgb_data is None or depth_data is None:
            continue

        results = model.predict(rgb_data)
        segmentation_mask = np.zeros_like(rgb_data, dtype=np.uint8)

        for result in results:
            for mask, cls_id in zip(result.masks.xy, result.boxes):
                points = np.int32([mask])
                color_number = classes_ids.index(int(cls_id.cls[0]))
                cv2.fillPoly(segmentation_mask, points, colors[color_number])

        segmentation_result = cv2.addWeighted(rgb_data, 1, segmentation_mask, 0.7, 0)

        # Display result
        cv2.imshow("WINDOW", segmentation_result.astype(np.uint8))

        # Handle user input
        msg = cv2.waitKey(1)
        if msg == ord("q") or msg & 0xFF == 27 or cv2.getWindowProperty('WINDOW', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Stop the sensor and close windows
    sensor.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
