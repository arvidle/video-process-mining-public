from .base import DetectionBasedTracker, DetectionResults
from .. import TrackingResults


class ExternalTracker(DetectionBasedTracker):
    def __init__(self, filename: str):
        self.filename = filename

    def run(self, detections: DetectionResults = None) -> TrackingResults:
        with open(self.filename, "r") as file:
            results = file.read()

        return TrackingResults(results)

