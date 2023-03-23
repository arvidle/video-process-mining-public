from video_pm import Video
from typing import Generator, Tuple
import numpy as np

class VideoVisualizer(Video):
    def __init__(self, video: Video):
        pass

    def frames(self) -> Generator[Tuple[bool, np.ndarray], None, None]:
        pass
