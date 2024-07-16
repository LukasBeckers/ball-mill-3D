import cv2
import logging
import numpy as np
from typing import List


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


class videoLoader():
    """
    Loads a video and stores the frames using "load_video" method.
    Frames can be accessed by indexing the instance of this class.
    """
    def __init__(self):
        self.cap = None
        self.index = 0
        pass

    def _load_frames(self) -> List[np.ndarray]:
        assert self.cap is not None, "Load a video first using the 'load_video' method."

        frames = []
        i = 0
        while (self.cap.isOpened()):
            i +=  1
            ret, frame = self.cap.read()
            if ret:
                frames.append(frame)
            if frame is None:
                break
        return frames

    def load_video(self, video_path: str):
        """
        Loads all frames from the video stored at "video_path",
        and stores the frames at self.frames
        """
        assert type(video_path) == str, "video_path must be of type string."

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            logging.error(f"Error opening video file {video_path}!")
            raise FileNotFoundError(f"Error opening video file {video_path}!")

        self.frames = np.array(self._load_frames())
        self.totalFrames = len(self.frames)

    def __getitem__(self, index: int) -> np.ndarray:
        assert type(index) == int, "index must be of type int."
        return self.frames[index]

    def __len__(self) -> int:
        return self.totalFrames


if __name__=="__main__":
    pass
