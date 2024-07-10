import cv2
import math
import numpy as np
import yolov5
from pyorbbecsdk import *
from utils import frame_to_bgr_image
import socket
import json

def pixel_to_camera_coordinates(u, v, d, fx, fy, cx, cy):
    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d
    return X, Y, Z

def calculate_3d_coordinates(u, v, W_image, W_real, D_known, fx, fy, cx, cy):
    Z = (D_known * W_real) / W_image
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y, Z

def send_data(data):
    host = '127.0.0.1'
    port = 65432
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            s.sendall(data.encode('utf-8'))
    except ConnectionRefusedError:
        print("Connection refused, please ensure the receiver is running")

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
        config.enable_stream(depth_profile)
    except Exception as e:
        print(e)
        return

    if device_pid == 0x066B:
        config.set_align_mode(OBAlignMode.SW_MODE)
    else:
        config.set_align_mode(OBAlignMode.HW_MODE)

    model = yolov5.load('./camera.pt')

    pipeline.start(config)
    camera_param = pipeline.get_camera_param()

    while True:
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            continue
        color_image = frame_to_bgr_image(color_frame)
        results = model(color_image, augment=True)
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
        depth_image = cv2.addWeighted(color_image, 0.5, depth_image, 0.5, 0)

        detected_objects = []

        for index, row in detections.iterrows():
            center_x = int((row['xmin'] + row['xmax']) / 2)
            center_y = int((row['ymin'] + row['ymax']) / 2)
            fx = 545.103
            fy = 545.103
            cx = 321.608
            cy = 243.754
            X, Y, Z = pixel_to_camera_coordinates(center_x, center_y, depth_data[center_y][center_x], fx, fy, cx, cy)

            W_real = 60
            D_known = 553
            X1, Y1, Z1 = calculate_3d_coordinates(center_x, center_y, (row['xmax'] - row['xmin']), W_real, D_known, fx, fy, cx, cy)

            depth_cv = math.sqrt(X1 * X1 + Y1 * Y1 + Z1 * Z1)
            if (1 - row["confidence"] > depth_cv * 0.0025):
                continue
            cv2.putText(depth_image, f"{int(X)},{int(Y)},{int(Z)}\n{int(X1)},{int(Y1)},{int(Z1)}", (center_x, center_y), 5, 1, (255, 255, 255))
            
            detected_objects.append({
                "class": row['name'],
                "confidence": float(row['confidence']),
                "coordinates": {
                    "center": [center_x, center_y],
                    "pixel_to_camera": [float(X), float(Y), float(Z)],
                    "calculated_3d": [float(X1), float(Y1), float(Z1)]
                }
            })

        if detected_objects:
            send_data(json.dumps(detected_objects))
        else:
            send_data(json.dumps([{'class': 'A', 'confidence': 0, 'coordinates': {'center': [0, 0],'calculated_3d': [10.0, 0.0, 1.0]}}]))

        cv2.imshow("SyncAlignViewer", depth_image)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
