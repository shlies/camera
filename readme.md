# 配置环境

## 克隆仓库
    
    git clone https://github.com/shlies/camera.git

## 创建虚拟环境

    conda create -n camera_env python=3.8
    conda activate camera_env
    conda install -c conda-forge gcc

## 安装依赖
### 直接安装

    cd ./camera
    pip install -r requirements.txt

### 或编译安装

下载仓库

    git clone https://github.com/orbbec/pyorbbecsdk.git
    sudo apt-get install python3-dev python3-venv python3-pip python3-opencv
    
If you use Anaconda, set the Python3 path to the Anaconda path in pyorbbecsdk/CMakeLists.txt before the find_package(Python3 REQUIRED COMPONENTS Interpreter Development) line:
    
    set(Python3_ROOT_DIR "/home/anaconda3/envs/py3.6.8") # Replace with your Python3 path
set(pybind11_DIR "${Python3_ROOT_DIR}/lib/python3.6/site-packages/pybind11/share/cmake/pybind11") # Replace with your Pybind11 path

Build the Project

   cd pyorbbecsdk
   pip3 install -r requirements.txt
    mkdir build
    cd build
    cmake -Dpybind11_DIR=`pybind11-config --cmakedir` ..
    make -j4
    make install
    cd ..
    pip3 install wheel
    python3 setup.py bdist_wheel
    pip3 install dist/*.whl
    pip3 install yolov5
    
Go!
    python color_viewer.py
