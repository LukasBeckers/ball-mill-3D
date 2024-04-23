import cv2
import numpy as np


class videoLoader():
    """
    Loads a video and stores the frames using "load_video" method.
    Frames can be accessed by indexing the instance of this class.
    """
    def __init__(self):
        self.cap = None
        self.index = 0
        pass

    def _load_frames(self):
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

    def load_video(self, video_path):
        """
        Loads all frames from the video stored at "video_path",
        and stores the frames at self.frames
        """
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Error opening video file {video_path}!")

        self.frames = np.array(self._load_frames())
        self.totalFrames = len(self.frames)

    def __getitem__(self, index):

        return self.frames[index]

    def __len__(self):
        return len(self.frames)


if __name__=="__main__":
    pass

