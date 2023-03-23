from video_pm.tracking.two_step import DetectionResults, concat_detection_results
DATA_DIR = "../../data/detection/"

training_results_filename = "../../data/detection/training_results/results_2.txt"

detection_days = [
    "ch01_20211113.npz",
    "ch01_20211114.npz",
    "ch01_20211115.npz",
    "ch01_20211116.npz",
    "ch01_20211117.npz"
]

detection_results_days_filenames = [DATA_DIR + day for day in detection_days]

detection_results_days = [DetectionResults.from_file(filename) for filename in detection_results_days_filenames]

all_detections = concat_detection_results(detection_results_days)
