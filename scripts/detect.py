import os
import shutil
import time

import cv2


def detect_video(model, src_video_path, dst_video_path, decode_fn, temp_frames='frames'):
    # 创建临时文件夹，用于存储测试过程中产生的临时文件
    if not os.path.exists(temp_frames):
        os.makedirs(temp_frames)

    # 读取原始视频
    video_capture = cv2.VideoCapture(src_video_path)

    # 码率
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    # 帧尺寸
    frame_size = (
        int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(dst_video_path, fourcc, fps, frame_size)

    while True:
        ret, frame = video_capture.read()
        if ret:
            t1 = time.time()
            cv2.imwrite(os.path.join(temp_frames, f"frame_temp.jpg"), frame)
            frame_dir = os.path.join(temp_frames, f"frame_temp.jpg")
            new_frame = decode_fn(model, frame_dir, print_on=False, save_result=False)
            t2 = time.time()
            fps = (1 / (t2 - t1) + fps) / 2
            # 加上帧率显示
            new_frame = cv2.putText(new_frame, f"fps= {fps:.2f}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # 让显示窗口可以调整大小
            cv2.namedWindow(f"detection_results", 0)
            cv2.imshow(f"detection_results", new_frame)
            # 保存视频帧
            video_writer.write(new_frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

    video_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()
    shutil.rmtree(temp_frames)