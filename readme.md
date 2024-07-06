# 配置环境

克隆仓库
    
    git clone https://github.com/shlies/camera.git

创建虚拟环境

    conda create -n camera_env python=3.8
    conda activate camera_env
    conda install -c conda-forge gcc

安装依赖

    cd ./camera
    pip install -r requirements.txt
    
Go!
    python color_viewer.py
