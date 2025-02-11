import torch
import torchvision

import cv2
import numpy as np
from rs_sensor import RSSensor  # Import the RSSensor class
from PIL import Image

# BiRefNet Repository
from models.birefnet import BiRefNet
from utils import check_state_dict

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

# MODEL_PATH = "../model/BiRefNet_HR-general-epoch_230.pth"
# MODEL_INPUT_SIZE = (2048, 2048)

MODEL_PATH = "../model/BiRefNet-general-epoch_244.pth"
MODEL_INPUT_SIZE = (1024, 1024)


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
    print(camera_matrix, camera_params)

    # Load model
    birefnet = BiRefNet(bb_pretrained=False)
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    state_dict = check_state_dict(state_dict)  # Check and process the state dictionary (상태 사전 확인 및 처리)
    birefnet.load_state_dict(state_dict)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision(
        ['high', 'highest'][0])  # Increase matrix multiplication precision (행렬 곱셈 정밀도 설정)
    birefnet.to(device)
    birefnet.eval()  # Set the model to evaluation mode

    # Define preprocessing transformations (전처리 변환 정의)
    transform_image = torchvision.transforms.Compose([
        torchvision.transforms.Resize(MODEL_INPUT_SIZE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    while True:
        # Retrieve sensor data
        rgb_data, depth_data = sensor.get_data()
        if rgb_data is None or depth_data is None:
            continue

        img_pil = Image.fromarray(rgb_data)
        input_images = transform_image(img_pil).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = torchvision.transforms.ToPILImage()(pred)  # 텐서를 PIL 이미지로 변환
        pred_pil = pred_pil.resize((rgb_data.shape[1], rgb_data.shape[0]))
        pred_np = np.array(pred_pil)

        # Threshold-based binarization
        threshold = 64
        binary_mask = (pred_np > threshold).astype(np.uint8)
        cv2.imshow("test", binary_mask * 255)

        # Apply mask
        color_mask = np.array([0, 255, 0])  # 녹색 (RGB)
        rgb_data[binary_mask == 1] = color_mask * 0.35 + rgb_data[binary_mask == 1] * 0.65

        # Display result
        cv2.imshow("WINDOW", rgb_data)

        # Handle user input
        msg = cv2.waitKey(1)
        if msg == ord("q") or msg & 0xFF == 27 or cv2.getWindowProperty('WINDOW', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Stop the sensor and close windows
    sensor.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
