from .detectron2_detector import Detectron2Detector
from .bytetrack_tracker import ByteTrackTracker
from .external_detector import ExternalDetector
from .base import TwoStepTracker


class Detectron2ByteTrackTracker(TwoStepTracker):
    def __init__(self):
        super().__init__(Detectron2Detector(), ByteTrackTracker())


class ExternalDetectorByteTrackTracker(TwoStepTracker):
    def __init__(self, dets_filename: str):
        super().__init__(ExternalDetector(dets_filename), ByteTrackTracker())
