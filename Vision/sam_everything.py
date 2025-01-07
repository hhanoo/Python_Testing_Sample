import torch
import torchvision

import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from rs_sensor import RSSensor  # Import the RSSensor class

torch.cuda.empty_cache()
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
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint="../model/sam_vit_h_4b8939.pth")
    sam.to(device=device)
    maskGenerator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,  # 포인트 그리드 한 변에 배치할 점의 수
        points_per_batch=64,  # 한 번에 처리할 포인트 개수
        pred_iou_thresh=0.88,  # IoU 임계값
        stability_score_thresh=0.98,  # 마스크 안정성 점수 임계값
        stability_score_offset=1.0,  # 안정성 점수를 계산할 때 사용 할 오프셋 값
        box_nms_thresh=0.3,  # 박스 간 비최대 억제(NMS) 임계값
        min_mask_region_area=500,  # 마스크 최소 영역(픽셀 수)
    )

    while True:
        # Retrieve sensor data
        rgb_data, depth_data = sensor.get_data()
        if rgb_data is None or depth_data is None:
            continue

        anns = maskGenerator.generate(rgb_data)

        if len(anns) == 0:
            continue

        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.random.random(3)
            rgb_data[m > 0] = color_mask * 0.35 + rgb_data[m > 0] * 0.65

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
