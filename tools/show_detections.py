from video_pm.tracking.two_step import ExternalDetector, ExternalYOLODetector, DetectionResults
from video_pm.visualization.tracking import DetectionVideo
from video_pm import Video
import cv2
import numpy as np

#FILENAME1 = "/home/arvid/Downloads/ch01_20211119094836_480_masked (2).txt"
FILENAME2 = "../data/detection/ch01_20211114061218.npz"
VIDEO = "/home/arvid/Documents/Arbeit_Data/videos/ch01_20211114/converted/ch01_20211114061218.mp4"

video = Video(VIDEO)

#detector1 = ExternalYOLODetector(FILENAME1)

#detections1 = detector1.run(video)
#detections1.filter(0.2, 0.8, 0.05)
detections2 = DetectionResults.from_file(FILENAME2)

#dv1 = DetectionVideo(detections1, video)
dv2 = DetectionVideo(detections2, video)

for frame in dv2.frames():
 #   frame = np.hstack((frame1, frame2))
    frame = cv2.resize(frame, (854, 480))
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
