## Environment

- OS: Ubuntu 20.04 LTS
- CUDA: 12.x
- Language: Python 3.10
- VirtualEnv: venv
- IDE: PyCharm
- Device: RealSense
- GPU : RTX 4090

## Necessary Install

- PIP Update
  ```bash
  pip install --upgrade pip
  ```

- RealSense
  ```bash
  pip install pyrealsense2
  pip install opencv-python
  ```

- Torch (Version: Linux, Pip, Python, CUDA 12.1)
  ```bash
  pip3 install torch torchvision torchaudio
  ```

## Required Install

- PyInstaller [[Manual Link](https://pyinstaller.org/en/stable/)]
  ```bash
  pip install pyinstaller
  ```
  
- Segment-anything (SAM) [[Manual Link](https://github.com/facebookresearch/segment-anything)]
  ```bash
  pip install git+https://github.com/facebookresearch/segment-anything.git
  ```

- Ultralytics (Yolo) [[Manual Link](https://github.com/ultralytics/ultralytics)]
  ```bash
  pip install ultralytics
  ```

- BiRefNet [[Manual Link](https://huggingface.co/ZhengPeng7/BiRefNet)]
  ```bash
  pip install timm
  pip install scikit-image
  pip install kornia
  pip install einops
  pip install prettytable
  pip install huggingface-hub
  pip install accelerate
  ```

- Llama [[Manual Link](https://github.com/meta-llama)]
  - Type 1: Using huggingface
    ```bash
    pip install --upgrade transformers
    pip install accelerate
    ```
  - Type 2: Using ollama
    ```bash
    pip install ollama
    ```
  
- Grounded SAM2 (In Grounded SAM folder) [[Manual Link](https://github.com/IDEA-Research/Grounded-SAM-2)]
  ```bash
  cd grounded_sam
  pip install -e .
  python -m pip install -e segment_anything
  pip install --no-build-isolation -e grounding_dino
  ```