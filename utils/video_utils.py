import cv2
import numpy as np


class videoLoader():
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
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("Error opening video file!")

        self.frames = np.array(self._load_frames())
        self.totalFrames = len(self.frames)


    def __getitem__(self, index):

        return self.frames[index]

    def __len__(self):
        return len(self.frames)


if __name__=="__main__":
    vL = videoLoader()
    vL.load_video("../videos/VID-20240329-WA0003.mp4", start_frame=100, end_frame=-100)
    for frame in vL:
        cv2.imshow("Window1", frame)
        cv2.waitKey(0)


