import torch
import torchvision

import cv2
import numpy as np
from rs_sensor import RSSensor  # Import the RSSensor class

import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import Model

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

"""
Hyper parameters
"""
TEXT_PROMPT = "red square. blue square. green square"
SAM2_CHECKPOINT = "../model/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "../../SDK/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "../model/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # Initialize the sensor
    sensor = RSSensor()
    sensor.get_device_sn()
    sensor.start()

    # Load SAM2 image predictor
    sam2_checkpoint = SAM2_CHECKPOINT
    model_cfg = SAM2_MODEL_CONFIG
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # Load Grounding DINO model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG,
                                 model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
                                 device=DEVICE)

    if_init = True

    while True:
        # Retrieve sensor data
        rgb_data, depth_data = sensor.get_data()
        if rgb_data is None or depth_data is None:
            continue

        if if_init:
            sam2_predictor.set_image(rgb_data)
            if_init = False

        # Grounding DINO로 객체 감지
        detections, labels = grounding_dino_model.predict_with_caption(
            image=rgb_data,
            caption=TEXT_PROMPT,
            box_threshold=0.40,
            text_threshold=0.25,
        )

        if len(detections) <= 0:
            cv2.imshow("WINDOW", rgb_data)

        elif len(detections) > 0:
            # Extract bounding boxes
            input_boxes = detections.xyxy

            if torch.cuda.get_device_properties(0).major >= 8:
                # Turn on tfloat32 for Ampere GPUs
                # (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                # Run SAM2 inference
                masks, scores, logits = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )

                # convert the shape to (n, H, W)
                if masks.ndim == 4:
                    masks = masks.squeeze(1)

                confidences = detections.confidence.tolist()
                class_names = labels
                class_ids = np.array(list(range(len(class_names))))

                img_labels = [
                    f"{class_name} {confidence:.2f} "
                    for class_name, confidence
                    in zip(class_names, confidences)
                ]

                detections = sv.Detections(
                    xyxy=input_boxes,  # (n, 4)
                    mask=masks.astype(bool),  # (n, h, w)
                    class_id=class_ids
                )

                box_annotator = sv.BoxAnnotator()
                annotated_frame = box_annotator.annotate(scene=Image.fromarray(rgb_data), detections=detections)

                label_annotator = sv.LabelAnnotator()
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=img_labels)

                cv2.imshow("WINDOW", np.asarray(annotated_frame))

        # Handle user input
        msg = cv2.waitKey(1)
        if msg == ord("q") or msg & 0xFF == 27 or cv2.getWindowProperty('WINDOW', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Stop the sensor and close windows
    sensor.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
