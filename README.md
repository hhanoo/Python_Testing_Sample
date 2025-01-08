# Python Testing Sample

## Environment

- OS: Ubuntu 20.04 LTS
- CUDA: 12.x
- Language: Python 3.10
- VirtualEnv: venv
- IDE: PyCharm
- Device: RealSense
- GPU : RTX 4090

## Install

- PIP Update
  ```bash
  pip install --upgrade pip
  ```
  
- RealSense
  ```bash
  pip install pyrealsense2
  ```
  
- PyInstaller ([Manual Link](https://pyinstaller.org/en/stable/))
  ```bash
  pip install pyinstaller
  ```

- Torch (Version: Linux, Pip, Python, CUDA 12.1)
  ```bash
  pip3 install torch torchvision torchaudio
  ```

- Segment-anything  (SAM)
  ```bash
  pip install git+https://github.com/facebookresearch/segment-anything.git
  ```

- Ultralytics (Yolo)
  ```bash
  pip install ultralytics
  ```

- BiRefNet ([Manual Link](https://huggingface.co/ZhengPeng7/BiRefNet))
  ```bash
  pip install timm
  pip install scikit-image
  pip install kornia
  pip install einops
  pip install prettytable
  pip install huggingface-hub
  pip install accelerate
  ```

- Llama ()
  ```bash
  pip install --upgrade transformers
  pip install accelerate
  ```