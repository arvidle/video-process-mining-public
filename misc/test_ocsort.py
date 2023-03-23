import os
from video_pm.tracking.two_step import OCSortTracker, DetectionResults

FILENAME = "ch01_20211114061218.npz"
DET_DIR = "../data/detection"
TRACK_DIR = "../data/tracking"
DET_FILENAME = os.path.join(DET_DIR, FILENAME)
TRACK_FILENAME = os.path.join(TRACK_DIR, FILENAME)


detections: DetectionResults = DetectionResults.from_file(DET_FILENAME)
#detections.detections = detections.detections[:50000]
#detections.filter(0.4, 0.9, 0.02)


tracker = OCSortTracker(det_thresh=0.48, iou_threshold=0.23)
#tracker = OCSortTracker.optimize(detections, 200)
results = tracker.run(detections)

print("Number of unique track IDs: ", len(results.tracking["track_id"].unique()))

#results.to_file(TRACK_FILENAME)

