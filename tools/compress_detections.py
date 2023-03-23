from video_pm.tracking.two_step import DetectionResults, ExternalYOLODetector
from video_pm import Video
import os

FILENAME = "/home/arvid/Downloads/ch01_20211113070651.txt"
VIDEO_FILENAME = "/home/arvid/Documents/Arbeit_Data/videos/ch01_20211113/converted/ch01_20211113070651.mp4"
OUTPUT = os.path.join("../data/detection", os.path.splitext(os.path.basename(FILENAME))[0] + ".npz")

video = Video(VIDEO_FILENAME)

detector = ExternalYOLODetector(FILENAME)
detections = detector.run(video)
detections.to_file(OUTPUT)
