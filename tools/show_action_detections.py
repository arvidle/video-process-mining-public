from video_pm.activity_recognition import ActionDetectionResults
from video_pm.visualization.activity_recognition import ActionDetectionVideo
from video_pm import Video
import cv2
import numpy as np

#FILENAME1 = "/home/arvid/Downloads/ch01_20211119094836_480_masked (2).txt"
FILENAME2 = "../data/action_detection/loaded/20211113.pkl"
VIDEO = "/home/arvid/Documents/Arbeit_Data/videos/ch01_20211113/converted/ch01_20211113.mp4"

video = Video(VIDEO)

#detector1 = ExternalYOLODetector(FILENAME1)

#detections1 = detector1.run(video)
#detections1.filter(0.2, 0.8, 0.05)
detections2 = ActionDetectionResults.from_file(FILENAME2)

#dv1 = DetectionVideo(detections1, video)
dv2 = ActionDetectionVideo(detections2, video)

fps = 1
width = 854 # * 2
height = 480
output_size = (width, height)
out = cv2.VideoWriter("action_detection.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, output_size)

for frame in dv2.frames(400000):
 #   frame = np.hstack((frame1, frame2))
    frame = cv2.resize(frame, (854, 480))
    cv2.imshow("Frame", frame)
    out.write(frame)
    cv2.waitKey(1)