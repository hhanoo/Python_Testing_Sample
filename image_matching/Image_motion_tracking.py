import cv2
import numpy as np
from rs_sensor import RSSensor  # Import the RSSensor class
from scipy.ndimage import gaussian_filter1d


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

    # Load base image (Initial Image)
    rgb_data, depth_data = sensor.get_data()
    base_img = rgb_data

    base_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)

    # ORB 초기화
    orb = cv2.ORB_create()
    keypoints_ref, descriptors_ref = orb.detectAndCompute(base_gray, None)

    # 초기화
    translations = []
    rotations = []
    scales = []

    while True:
        # Retrieve sensor data
        rgb_data, depth_data = sensor.get_data()
        if rgb_data is None or depth_data is None:
            continue

        current_gray = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2GRAY)

        # 현재 이미지의 특징점 추출
        keypoints_cur, descriptors_cur = orb.detectAndCompute(current_gray, None)

        # 특징점 매칭 (Brute-Force Matcher)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors_ref, descriptors_cur)
        matches = sorted(matches, key=lambda x: x.distance)

        # 상위 매칭점만 사용
        num_matches = 10
        matches = matches[:num_matches]

        # 정답 이미지와 현재 이미지의 매칭된 특징점 좌표 추출
        points_ref = np.float32([keypoints_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        points_cur = np.float32([keypoints_cur[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 변환 행렬 계산 (Affine or Homography)
        matrix, mask = cv2.estimateAffinePartial2D(points_ref, points_cur)

        # 이동 거리 및 매칭된 특징점 이동 표시
        distances = []
        for (p_ref, p_cur) in zip(points_ref, points_cur):
            x1, y1 = p_ref.ravel()
            x2, y2 = p_cur.ravel()
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distances.append(distance)

            # 특징점 연결 그리기
            cv2.line(rgb_data, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(rgb_data, (int(x2), int(y2)), 5, (0, 0, 255), -1)

        if matrix is not None:
            # 이동 벡터 추출
            dx, dy = matrix[0, 2], matrix[1, 2]
            # 회전 각도 계산
            angle = np.arctan2(matrix[1, 0], matrix[0, 0]) * (180.0 / np.pi)
            # 스케일 변화 추정
            scale = np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)

            # 값 추가
            translations.append((dx, dy))
            rotations.append(angle)
            scales.append(scale)

            if len(translations) > 10:  # 최근 10개의 데이터만 유지
                translations.pop(0)
                rotations.pop(0)
                scales.pop(0)

            # 필터 적용
            smoothed_dx = gaussian_filter1d([t[0] for t in translations], sigma=2)[-1]
            smoothed_dy = gaussian_filter1d([t[1] for t in translations], sigma=2)[-1]
            smoothed_angle = gaussian_filter1d(rotations, sigma=2)[-1]
            smoothed_scale = gaussian_filter1d(scales, sigma=2)[-1]

            print(
                f"Translation: dx={smoothed_dx:.2f}, dy={smoothed_dy:.2f}, Rotation: {smoothed_angle:.2f} degrees, Scale: {smoothed_scale:.2f}")

        # 매칭 시각화 - 이미지를 겹쳐서 보기
        h, w, _ = base_img.shape
        overlay = cv2.addWeighted(base_img, 0.5, rgb_data, 0.5, 0)
        for (p_ref, p_cur) in zip(points_ref, points_cur):
            x1, y1 = map(int, p_ref.ravel())
            x2, y2 = map(int, p_cur.ravel())
            cv2.circle(overlay, (x1, y1), 5, (255, 0, 0), -1)  # 정답 이미지 특징점
            cv2.circle(overlay, (x2, y2), 5, (0, 255, 0), -1)  # 현재 이미지 특징점

        cv2.imshow("Overlay", overlay)

        # Handle user input
        msg = cv2.waitKey(1)
        if msg == ord("q") or msg & 0xFF == 27 or cv2.getWindowProperty('Overlay', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Stop the sensor and close windows
    sensor.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
