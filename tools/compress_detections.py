from video_pm.tracking.two_step import DetectionResults, ExternalYOLODetector
from video_pm import Video
import os

FILENAME = "../data/detection/yolov7_out_txt/954-0000-0001.txt"
VIDEO_FILENAME = "../data/videos/action_dataset/cropped_1m/video_crop/954-0000-0001.mp4"
OUTPUT = os.path.join("../data/detection/yolov7_out", os.path.splitext(os.path.basename(FILENAME))[0] + ".npz")

video = Video(VIDEO_FILENAME)

detector = ExternalYOLODetector(FILENAME)
detections = detector.run(video)
detections.to_file(OUTPUT)
