# Define Anaylsis
import numpy as np
import cv2


class VideoArea:
    def __init__(self, area):
        self.area: np.ndarray = area

    def __contains__(self, point):
        return bool(self.area[point])

    @classmethod
    def from_file(cls, filename):
        area = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        return cls(area)
