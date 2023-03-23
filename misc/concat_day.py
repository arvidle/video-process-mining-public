from video_pm.tracking.two_step import DetectionResults, concat_detection_results

det_filenames = [
"ch01_20211113055954",
"ch01_20211113070651",
"ch01_20211113084201",
"ch01_20211113102023",
"ch01_20211113120236",
"ch01_20211113133513",
"ch01_20211113150710",
"ch01_20211113163517",
]

data_dir = "../data/detection/"

paths = [data_dir + fname + ".npz" for fname in det_filenames]

detections = [DetectionResults.from_file(filename) for filename in paths]

whole_day_detections = concat_detection_results(detections)

whole_day_detections.to_file("../data/detection/ch01_20211113.npz")
