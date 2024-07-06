# ******************************************************************************
#  Copyright (c) 2023 Orbbec 3D Technology, Inc
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.  
#  You may obtain a copy of the License at
#  
#      http:# www.apache.org/licenses/LICENSE-2.0
#  
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************
import cv2,time

from pyorbbecsdk import Config
from pyorbbecsdk import OBError
from pyorbbecsdk import OBSensorType, OBFormat
from pyorbbecsdk import Pipeline, FrameSet
from pyorbbecsdk import VideoStreamProfile
import yolov5
# from pyorbbecsdk import *
from utils import frame_to_bgr_image

ESC_KEY = 27


def main():
    config = Config()
    pipeline = Pipeline()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
            print("color profile: ", color_profile)
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return
    print("\n\n\n\n\n\n\n\nLoding Model\n\n")
    model = yolov5.load('/home/shlies/yolov5/camera.pt')
    # cpkt=torch.load("./photos/camera.pt")
    # yolov5_load=model
    # yolov5_load.model=cpkt["model"]
    print("\n\n\n\n\nLoad complete\n\n")
    pipeline.start(config)
    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            # covert to RGB format
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                continue
            results=model(color_image,augment=True)
            results.print()  # 打印结果到控制台
            # results.show()   # 显示结果图像
            results.render()

            cv2.imshow("Color Viewer", color_image)
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break
            #按下t键保存图片
            date=str(time.time())

            if key == ord('t'):
                cv2.imwrite("photos/color_image"+date+".jpg", color_image)
        except KeyboardInterrupt:
            break
    pipeline.stop()


if __name__ == "__main__":
    main()
