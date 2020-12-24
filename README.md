# Check-in system based on Face Recognition

## Introduction

In this project, we develop and deploy a **check-in system based on Face Recognition** on **Jetson Nano** embedded computer.

![alt text](https://github.com/ngoductuanlhp/FRCheckInSystem/blob/main/images/jetson_nano.jpeg?raw=false)


## Requirements

### Hardware

1. Jetson Nano 4GB
2. CSI camera / Orbbec Astra Camera (USB)

### Software

1. OS: **Jetpack 4.4.1** or later (We do not ensure that the older Jetpack will work well without problems)
2. OpenCV 4.1, cuDNN 8.0, Cuda 10.2, TensorRT 7.1.3 (all are included in Jetpack 4.4.1)
3. Python package:
    * pycuda
    * tensorrt
    * pickle
    * filterpy
    * numpy
    * scipy (we recommend you install scipy following NVidia's intruction on Jetson Nano to avoid any hardware conflict)

## Weights

Link download: [weights](https://drive.google.com/drive/folders/1eC72Su4MwZJ67OdsCyEaS3VUPV42fb1l?usp=sharing)

## Instructions

1. Install Jetpack 4.4.1 into Jetson Nano based on the instruction from NVidia
2. Clone the repo
3. Install all packages listed in requirements.txt 

    ```bash
    # Install with pip
    pip3 install -r requirements.txt
    ```

4. If any package conflicts with the hardware, please search in Google or NVidia forum for solution :))).
5. Download ONNX weight for FaceDetection (in our provided link), we provide 3 models for different resolutions for the input image. Put it in folder weights (create if not exist). Modify the name of onnx model and trt model in **onnx_to_trt.py**. Run the following command to build the TensorRT engine (it can take a few minutes). Put the new TRT engine to the engines folder.

    ```bash
    python3 onnx_to_trt.py
    ```
    
6. Download ONNX weight for FaceIdentifier (in our provided link). Put it in folder weights (create if not exist). Modify the name of onnx model and trt model in **onnx_to_trt.py**. Run the following command to build the TensorRT engine (it can take a few minutes). Put the new TRT engines to the engines folder.

    ```bash
    python3 onnx_to_trt.py
    ```

7. Test the camera:

    7.1 CSI camera: 

    * Plug into the CSI slot, run the following command to test the camera:

    ```bash
    #(Crtl-C to exit)
    gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=3820, height=2464, framerate=21/1, format=NV12' ! nvvidconv flip-method=0 ! 'video/x-raw,width=960, height=616' ! nvvidconv ! nvegltransform ! nveglglessink -e
    ```

    * If there is any error during executing the above command, reset the hardware by typing:

    ```bash
    adasd
    ```

    * If it does not work, please check again your camera.

    7.2 Orbbec Astra camera:
    * Plug into the USB slot, run:

    ```bash
    # switch to super user
    sudo -s
    # test camera (Crtl-C to exit)
    python3 test_camera.py
    # exit super user
    exit
    ```

## Demo

### Demo with video (only detection, no identification)

1. Put the video file in folder videos.
2. Run:

    ```bash
    python3 demo_video.py --vid video_name.mp4
    ```

### Demo with real camera

Run:

```bash
python3 demo_camera.py --csicam True --orbcam True
```

