# yolov5

The Pytorch implementation is [ultralytics/yolov5](https://github.com/ultralytics/yolov5).

## Different versions of yolov5

Currently, we support yolov5 v1.0(yolov5s only), v2.0, v3.0, v3.1, v4.0 and v5.0.

- For yolov5 v5.0, download .pt from [yolov5 release v5.0](https://github.com/ultralytics/yolov5/releases/tag/v5.0), `git clone -b v5.0 https://github.com/ultralytics/yolov5.git` and `git clone https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in current page.
- For yolov5 v4.0, download .pt from [yolov5 release v4.0](https://github.com/ultralytics/yolov5/releases/tag/v4.0), `git clone -b v4.0 https://github.com/ultralytics/yolov5.git` and `git clone -b yolov5-v4.0 https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in [tensorrtx/yolov5-v4.0](https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v4.0/yolov5).
- For yolov5 v3.1, download .pt from [yolov5 release v3.1](https://github.com/ultralytics/yolov5/releases/tag/v3.1), `git clone -b v3.1 https://github.com/ultralytics/yolov5.git` and `git clone -b yolov5-v3.1 https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in [tensorrtx/yolov5-v3.1](https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v3.1/yolov5).
- For yolov5 v3.0, download .pt from [yolov5 release v3.0](https://github.com/ultralytics/yolov5/releases/tag/v3.0), `git clone -b v3.0 https://github.com/ultralytics/yolov5.git` and `git clone -b yolov5-v3.0 https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in [tensorrtx/yolov5-v3.0](https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v3.0/yolov5).
- For yolov5 v2.0, download .pt from [yolov5 release v2.0](https://github.com/ultralytics/yolov5/releases/tag/v2.0), `git clone -b v2.0 https://github.com/ultralytics/yolov5.git` and `git clone -b yolov5-v2.0 https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in [tensorrtx/yolov5-v2.0](https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v2.0/yolov5).
- For yolov5 v1.0, download .pt from [yolov5 release v1.0](https://github.com/ultralytics/yolov5/releases/tag/v1.0), `git clone -b v1.0 https://github.com/ultralytics/yolov5.git` and `git clone -b yolov5-v1.0 https://github.com/wang-xinyu/tensorrtx.git`, then follow how-to-run in [tensorrtx/yolov5-v1.0](https://github.com/wang-xinyu/tensorrtx/tree/yolov5-v1.0/yolov5).

## Config

- Choose the model s/m/l/x/s6/m6/l6/x6 from command line arguments.
- Input shape defined in yololayer.h
- Number of classes defined in yololayer.h, **DO NOT FORGET TO ADAPT THIS, If using your own model**
- INT8/FP16/FP32 can be selected by the macro in yolov5.cpp, **INT8 need more steps, pls follow `How to Run` first and then go the `INT8 Quantization` below**
- GPU id can be selected by the macro in yolov5.cpp
- NMS thresh in yolov5.cpp
- BBox confidence thresh in yolov5.cpp
- Batch size in yolov5.cpp

## How to Run, yolov5s as example

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```
git clone -b v5.0 https://github.com/ultralytics/yolov5.git
git clone https://github.com/wang-xinyu/tensorrtx.git
// download https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt
cp {tensorrtx}/yolov5/gen_wts.py {ultralytics}/yolov5
cd {ultralytics}/yolov5
python gen_wts.py -w yolov5s.pt -o yolov5s.wts
// a file 'yolov5s.wts' will be generated.
```

2. build tensorrtx/yolov5 and run

```
cd {tensorrtx}/yolov5/
// update CLASS_NUM in yololayer.h if your model is trained on custom dataset
mkdir build
cd build
cp {ultralytics}/yolov5/yolov5s.wts {tensorrtx}/yolov5/build
cmake ..
make
sudo ./yolov5 -s [.wts] [.engine] [s/m/l/x/s6/m6/l6/x6 or c/c6 gd gw]  // serialize model to plan file
sudo ./yolov5 -d [.engine] [image folder]  // deserialize and run inference, the images in [image folder] will be processed.
// For example yolov5s
sudo ./yolov5 -s yolov5s.wts yolov5s.engine s
sudo ./yolov5 -d yolov5s.engine ../samples
// For example Custom model with depth_multiple=0.17, width_multiple=0.25 in yolov5.yaml
sudo ./yolov5 -s yolov5_custom.wts yolov5.engine c 0.17 0.25
sudo ./yolov5 -d yolov5.engine ../samples
```

3. check the images generated, as follows. _zidane.jpg and _bus.jpg

4. optional, load and run the tensorrt model in python

```
// install python-tensorrt, pycuda, etc.
// ensure the yolov5s.engine and libmyplugins.so have been built
python yolov5_trt.py
```

# INT8 Quantization

1. Prepare calibration images, you can randomly select 1000s images from your train set. For coco, you can also download my calibration images `coco_calib` from [GoogleDrive](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) pwd: a9wh

2. unzip it in yolov5/build

3. set the macro `USE_INT8` in yolov5.cpp and make

4. serialize the model and test

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247927-4d9fac00-751e-11ea-8b1b-704a0aeb3fcf.jpg">
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247970-60b27c00-751e-11ea-88df-41473fed4823.jpg">
</p>

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)

# 최종제출 결과 (데이터, 성능측정)

- 데이터 종류
    
      - dna_dataset : DNA대회 데이터셋
    
      - dna_dataset_v2_public : DNA대회 테스트용 데이터
    
      - sim_dataset : sim2data 가상데이터
    
      - sim_dataset_v2 : sim2data 가상데이터
    
      - sim_dataset_v3 : sim2data 가상데이터
    
     - aihub_dataset : AIhub 조난자 데이터 셋 30m이상
    
     - aihub_dataset_v2_25under : AI Hub 조난자 25m 이하 셋
    
     - aihub_dataset_v3_val : AI Hub 조난자 validation에서 추출
    
     - SARD_v1~6 : Search and Rescue Dataset (IEEE Dataport) 
    

- 데이터 학습

       - YOLOv5m 모델 사용

       - 초기 DNA 데이터와 SIM 가상데이터 조합으로 학습 한뒤, 

         실제 데이터 확보하여 추가 Transfer Learning 기법으로 성능 향상

       - 학습코드 ($home/dnadrone24/src/yolov5)

          : python [train.py](http://train.py) --img 640 --cfg yolov5m.yaml --hyp hyp.scratch.yaml --batch 32 --epochs 300 --data custom.yaml --weights [yolov5m.pt](http://yolov5s.pt/)

       - 정답체크  ($home/dnadrone24/src/yolov5_dna/)

          : python answer_sheet_creator.py --weights /home/dnadrone24/src/yolov5_dna/test_pt/final/final.pt --domain /home/dnadrone24/datasets_rw/challenge-dataset/public

- 속도 향상방법

      - TensorRTx 설치 후 YOLO모델을 TensorRT Engine 모델로 변환 

         ($home/dnadrone24/src/yolov5_dna/test_pt/final/final.engine)

         이미지 장당 속도를 측정하여 평균값 측정속도 제출

# **Install the dependencies of tensorrtx**

github : [https://github.com/wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)

## **Ubuntu**

Ubuntu18.04 / cuda10.0 / cudnn7.6.5 / tensorrt7.0.0 / opencv3.

### **1. Install CUDA**

Go to [cuda-10.0-download](https://developer.nvidia.com/cuda-10.0-download-archive). Choose `Linux` -> `x86_64` -> `Ubuntu` -> `18.04` -> `deb(local)` and download the .deb package.

Then follow the installation instructions.

`sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda`

### **2. Install TensorRT**

Go to [nvidia-tensorrt-7x-download](https://developer.nvidia.com/nvidia-tensorrt-7x-download). You might need login.

Choose TensorRT 7.0 and `TensorRT 7.0.0.11 for Ubuntu 1604 and CUDA 10.0 DEB local repo packages`

Install with following commands, after `apt install tensorrt`, it will automatically install cudnn, nvinfer, nvinfer-plugin, etc.

`sudo dpkg -i nv-tensorrt-repo-ubuntu1604-cuda10.0-trt7.0.0.11-ga-20191216_1-1_amd64.deb
sudo apt update
sudo apt install tensorrt`

추가 설치 명령

`sudo apt-get install libnvinfer8 libnvonnxparsers8 libnvparsers8 libnvinfer-plugin8`

`sudo apt-get install libnvinfer-dev libnvonnxparsers-dev libnvparsers-dev libnvinfer-plugin-dev`

`sudo apt-get install python3-libnvinfer`

### **3. Install OpenCV**

[https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/](https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/)

---

1. Uninstall the currently-install opencv with:
    
    `sudo apt remove libopencv-dev`
    
2. Refresh the packages index and install the OpenCV package by typing:
    
    `sudo apt update`
    
    `sudo apt install python3-opencv`
    
3. Install the required dependencies:
    
    `sudo apt install build-essential cmake git pkg-config libgtk-3-dev \    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \    gfortran openexr libatlas-base-dev python3-dev python3-numpy \    libtbb2 libtbb-dev libdc1394-22-dev`
    
4. Clone the OpenCV’s and OpenCV contrib repositories:
    
    `mkdir ~/opencv_build && cd ~/opencv_build`
    
    `git clone [https://github.com/opencv/opencv.git](https://github.com/opencv/opencv.git)`
    
    `git clone [https://github.com/opencv/opencv_contrib.git](https://github.com/opencv/opencv_contrib.git)`
    
5. Once the download is complete, create a temporary build directory, and switch to it:
    
    `cd ~/opencv_build/opencv`
    
    `mkdir build && cd build`
    
    Set up the OpenCV build with CMake:
    
    `cmake -D CMAKE_BUILD_TYPE=RELEASE \    -D CMAKE_INSTALL_PREFIX=/usr/local \    -D INSTALL_C_EXAMPLES=ON \    -D INSTALL_PYTHON_EXAMPLES=ON \    -D OPENCV_GENERATE_PKGCONFIG=ON \    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \    -D BUILD_EXAMPLES=ON ..`
    
6. Start the compilation process:
    
    `make -j8`
    
7. Install OpenCV with:
    
    `sudo make install`
    
8. Now do:
    
    `sudo ldconfig`
    

### **4. Check your installation**

`dpkg -l | grep cuda
dpkg -l | grep nvinfer
dpkg -l | grep opencv`

### **5. Run tensorrtx**

**How to Run, yolov5s as example**

`git clone -b v5.0 https://github.com/ultralytics/yolov5.git
git clone https://github.com/wang-xinyu/tensorrtx.git`

제출한 .pt 파일이 있는 경로로 `gen_wts.py` 를 복사-붙여넣기 시켜준다.

`cp {tensorrtx}/yolov5/gen_wts.py {ultralytics}/yolov5
cd {ultralytics}/yolov5
python gen_wts.py -w final.pt -o final.wts
// a file 'yolov5s.wts' will be generated.`

1. build tensorrtx/yolov5 and run

`cd {tensorrtx}/yolov5/
yololayer.h` 파일에 있는 CLASS_NUM을 1로 변경해준다.
`mkdir build
cd build`

만든 .wts 파일을 `{tensorrtx}/yolov5/build` 여기 경로로 복붙해준다.
`cp {ultralytics}/yolov5/final.wts {tensorrtx}/yolov5/build
cmake ..
make`

make에서 오류가 날 경우

tensorrtx/yolov5/build/CMakeList.txt 수정 → tensorrt version 문제

`include_directories(/home/dnadrone24/TensorRT-7.0.0.11/include) #for NvInfer.h`

`link_directories(/home/dnadrone24/TensorRT-7.0.0.11/lib/) #for cannot found -lnvinfer`

여기서 

`dpkg -L libnvinfer-dev` 출력 

```
/.
/usr
/usr/lib
/usr/lib/x86_64-linux-gnu
/usr/lib/x86_64-linux-gnu/libnvinfer_static.a
/usr/lib/x86_64-linux-gnu/libmyelin_compiler_static.a
/usr/lib/x86_64-linux-gnu/libmyelin_executor_static.a
/usr/lib/x86_64-linux-gnu/libmyelin_pattern_library_static.a
/usr/lib/x86_64-linux-gnu/libmyelin_pattern_runtime_static.a
/usr/include
/usr/include/x86_64-linux-gnu
/usr/include/x86_64-linux-gnu/NvInfer.h
/usr/include/x86_64-linux-gnu/NvInferRuntime.h
/usr/include/x86_64-linux-gnu/NvInferRuntimeCommon.h
/usr/include/x86_64-linux-gnu/NvInferVersion.h
/usr/include/x86_64-linux-gnu/NvUtils.h
/usr/share
/usr/share/doc
/usr/share/doc/libnvinfer-dev
/usr/share/doc/libnvinfer-dev/copyright
/usr/share/doc/libnvinfer-dev/changelog.Debian
/usr/lib/x86_64-linux-gnu/libmyelin.so
/usr/lib/x86_64-linux-gnu/libnvinfer.so
```

위 화면 출력되는 경우 정상

wts 파일과 저장하려는 engine 이름 넣고 모델 설정.
`sudo ./yolov5 -s [.wts] [.engine] [s/m/l/x/s6/m6/l6/x6 or c/c6 gd gw]  // serialize model to plan file
sudo ./yolov5 -d [.engine] [image folder]  // deserialize and run inference, the images in [image folder] will be processed.
// For example yolov5m
sudo ./yolov5 -s final.wts final.engine m`

test용 이미지들을 tensorrtx/yolov5/samples에 넣고 아래 코드 실행
`sudo ./yolov5 -d final.engine home/dnadrone24/datasets_rw/challenge-dataset/public`

