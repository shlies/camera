# coding=utf-8
'''
@file      main.py
@brief     主程序
@author    shlies (shlies@github.com)
@version   1.0.0
@date      2024-07-06

@copyright Copyright (c) 2024 Shlies. All Rights Reserved.

@attention 

@par history
| Version | Date | Author | Description |
| :---: | :---: | :---: | :---: |
| 1.0.0 | 2024-MM-DD | shlies | description |
@par last editor  shlies (shlies@github.com)
@par last edit time  2024-07-06
'''
import cv2
import numpy as np
import yolov5
import translationClass

from pyorbbecsdk import *
from utils import frame_to_bgr_image

def pixel_to_camera_coordinates(u, v, d, fx, fy, cx, cy):
    # 计算相机坐标系下的三维坐标
    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d
    return X, Y, Z
def calculate_3d_coordinates(u, v, W_image, W_real, D_known, fx, fy, cx, cy):
    # 计算深度 Z
    Z = (D_known * W_real) / W_image
    
    # 计算相机坐标系下的三维坐标
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    return X, Y, Z

def main():
    pipeline = Pipeline()
    device = pipeline.get_device()
    device_info = device.get_device_info()
    device_pid = device_info.get_pid()
    config = Config()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        assert profile_list is not None
        depth_profile = profile_list.get_default_video_stream_profile()
        assert depth_profile is not None
        print("color profile : {}x{}@{}_{}".format(color_profile.get_width(),
                                                   color_profile.get_height(),
                                                   color_profile.get_fps(),
                                                   color_profile.get_format()))
        print("depth profile : {}x{}@{}_{}".format(depth_profile.get_width(),
                                                   depth_profile.get_height(),
                                                   depth_profile.get_fps(),
                                                   depth_profile.get_format()))
        config.enable_stream(depth_profile)
    except Exception as e:
        print(e)
        return
    

    if device_pid == 0x066B:
        # Femto Mega does not support hardware D2C, and it is changed to software D2C
        config.set_align_mode(OBAlignMode.SW_MODE)
    else:
        config.set_align_mode(OBAlignMode.HW_MODE)

    print("\n\n\n\n\n\n\n\nLoding Model\n\n")
    model = yolov5.load('./camera.pt')
    print("\n\n\n\n\nLoad complete\n\n")

    translate = translationClass.Translation()
    

    # pipeline.enable_frame_sync()
    pipeline.start(config)
    camera_param = pipeline.get_camera_param()
    print(f"\n\n\n{camera_param}\n\n\n")
    while True:
        frames: FrameSet = pipeline.wait_for_frames(100)
        if frames is None:
            continue
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            continue
        color_image = frame_to_bgr_image(color_frame)
        results=model(color_image,augment=True)
        results.print()
        results.render()
        detections = results.pandas().xyxy[0]
        
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        scale = depth_frame.get_depth_scale()
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((height, width))
        depth_data = depth_data.astype(np.float32) * scale
        depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        # overlay color image on depth image
        depth_image = cv2.addWeighted(color_image, 0.5, depth_image, 0.5, 0)

        for index, row in detections.iterrows():
            print(f"index: {index} Class: {row['name']}, Confidence: {row['confidence']}, Coordinates: ({row['xmin']}, {row['ymin']}, {row['xmax']}, {row['ymax']})")
            if(row['confidence']<0.7):
                continue
            center_x=int((row['xmin']+row['xmax'])/2)
            center_y=int((row['ymin']+row['ymax'])/2)
            center=[center_x,center_y]
            fx = 545.103
            fy = 545.103
            cx = 321.608
            cy = 243.754
            X, Y, Z = pixel_to_camera_coordinates(center_x, center_y, depth_data[center_y][center_x], fx, fy, cx, cy)
            
            fx = 545.103
            fy = 545.103
            cx = 321.608
            cy = 243.754

            # 目标在已知距离下的实际大小
            W_real = 60  # 目标的宽度
            D_known = 553  # 已知距离

            # 计算三维坐标
            X1, Y1, Z1= calculate_3d_coordinates(center_x, center_y, (row['xmax']-row['xmin']), W_real, D_known, fx, fy, cx, cy)


            # translate.trans(center,depth_data)
            cv2.putText(depth_image,f"{int(X)},{int(Y)},{int(Z)}\n{int(X1)},{int(Y1)},{int(Z1)}", center,5,1,(255, 255, 255))
            print(f"position: {center_x},{center_y},depth={depth_data[center_y][center_x]} location: X={X}, Y={Y}, Z={Z} location_p: X={X1}, Y={Y1}, Z={Z1}")
        cv2.imshow("SyncAlignViewer ", depth_image)
        key = cv2.waitKey(1)

if __name__ == "__main__":
    main()