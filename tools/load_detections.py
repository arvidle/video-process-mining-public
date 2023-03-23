import os
from video_pm.tracking.two_step import TwoStepTracker, ExternalDetector, ByteTrackTracker, DetectionResults

external_detector = ExternalDetector(filename=os.path.abspath("../data/tracking_3_output.txt"))
tracker = ByteTrackTracker()

results = external_detector.run(video=None)
results.to_file("../data/detections.npz")

results2 = DetectionResults.from_file("../data/detections.npz")

print(results.detections.equals(results2.detections))

#detector = ExternalDetector("../data/tracking_3_output.txt")
#detections = detector.run()
#tracker = ByteTrackTracker.optimize(detections)
#tracks = tracker.run(detections)
#print(tracks)

