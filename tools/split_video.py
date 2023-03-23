from video_pm import Video
from video_pm.tracking import VideoIsolator, TrackingResults, TrackingResultsPostProcessor
import cv2
import os
from tqdm import tqdm

FILENAME = "ch01_20211119115931_480_masked"
TRACKING_EXT = ".npz"
VIDEO_EXT = ".mp4"
TRACKING_DIR = "tracking"
VIDEO_DIR = "videos"
DATA_PATH = "../data"
ISOLATED_DIR = "isolated"

VIDEO_PATH = os.path.join(DATA_PATH, VIDEO_DIR, FILENAME + VIDEO_EXT)
TRACKING_PATH = os.path.join(DATA_PATH, TRACKING_DIR, FILENAME + TRACKING_EXT)

video = Video(VIDEO_PATH)
tracking = TrackingResults.from_file(TRACKING_PATH)

post_processor = TrackingResultsPostProcessor(tracking)
processed_tracking = post_processor.run()

isolator = VideoIsolator(video, processed_tracking, group_on="trace_id")

isolated = isolator.run()


for isolated_video in tqdm(isolated, total=len(isolator)):
    isolated_filepath = FILENAME + "_" + str(isolated_video.tracking_trace.iloc[0]["trace_id"])
    isolated_filename = isolated_filepath + ".mp4"
    isolated_tracking_filename = isolated_filepath + ".csv"
    isolated_path = os.path.join(DATA_PATH, ISOLATED_DIR, isolated_filename)
    isolated_tracking_path = os.path.join(DATA_PATH, ISOLATED_DIR, isolated_tracking_filename)

    isolated_video.tracking_trace.to_csv(isolated_tracking_path)
    fps = 18.75
    width = isolated_video.width
    height = isolated_video.height
    output_size = (width, height)
    out = cv2.VideoWriter(isolated_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, output_size)
    for frame in isolated_video.frames():
        out.write(frame)
        #cv2.imshow("Frame", frame)
        #cv2.waitKey(1)

    out.release()
