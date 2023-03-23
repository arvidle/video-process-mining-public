import os
from video_pm.tracking.two_step import ByteTrackTracker, DetectionResults
import sys

files = ["ch01_20211113.npz", "ch01_20211114.npz", "ch01_20211115.npz", "ch01_20211116.npz", "ch01_20211117.npz"]

#FILENAME = "ch01_20211116.npz"
DET_DIR = "../data/detection"
TRACK_DIR = "../data/tracking/bytetrack"

for FILENAME in files:
    DET_FILENAME = os.path.join(DET_DIR, FILENAME)
    TRACK_FILENAME = os.path.join(TRACK_DIR, FILENAME)


    detections: DetectionResults = DetectionResults.from_file(DET_FILENAME)
    detections.filter(0.4, 0.9, 0.02)
    #detections.detections = detections.detections[:100000]

    tracker = ByteTrackTracker(0.75, 392, 0.83, frame_rate=30)
    #tracker = ByteTrackTracker.optimize(detections, 100)
    results = tracker.run(detections)

    print("Number of unique track IDs: ", len(results.tracking["track_id"].unique()))

    results.to_file(TRACK_FILENAME)

