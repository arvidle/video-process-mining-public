"""Video PM base module

Provides the base class for Videos
"""

import cv2
from typing import Generator, Tuple

import numpy as np

# Define Video


class Video:
    """Container class for video sequences

    """
    def __init__(self, filename: str):
        self.video: cv2.VideoCapture = None
        self.filename: str = filename
        self.reset()

    def __str__(self):
        return f"Video from {self.filename}"

    def frames(self) -> Generator[Tuple[bool, np.ndarray], None, None]:
        """Generator for iterating over the video frames

        :return: A generator over the video frames
        """
        self.reset()
        success, img = self.video.read()
        while success:
            yield success, img
            success, img = self.video.read()

    def reset(self) -> None:
        """Reset the internal video reader.

        :return: None
        """
        self.video = self.get_capture()

    def get_capture(self) -> cv2.VideoCapture:
        """Create a VideoCapture from the input video file.

        :return: A VideoCapture on the input video file.
        """
        return cv2.VideoCapture(self.filename)

