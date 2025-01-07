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
  #### My case: Driver 555.42, CUDA 12.5, ####
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
