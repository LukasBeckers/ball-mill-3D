from abc import ABC, abstractmethod
import numpy as np
import cv2

from camera import ICameraFrameProvider, Camera, NotCalibratedError


class IFrameProviderCalibrationDataManager(ABC):
    @abstractmethod
    def get_anchor_point(self, name: str) -> np.ndarray:
        pass

    @abstractmethod
    def save_anchor_point(self, name: str, anchor_point: np.ndarray):
        pass


class IFrameProviderGUI(ABC):
    @abstractmethod
    def send_message(self, image: np.ndarray, messsage: str):
        pass

    @abstractmethod
    def recieve_message(self) -> np.ndarray:
        pass


class CameraFrameProvider(ICameraFrameProvider):
    def __init__(self, camera: Camera, frame_provider_calibration_data_manager: IFrameProviderCalibrationDataManager, gui_interface: IFrameProviderGUI):
        self.camera = camera 
        self._i = 0
        self._frames = []
        self.calib_data_manager = frame_provider_calibration_data_manager
        self.gui_interface = gui_interface

    def _load_frames(self, video_dir: str):

        cap = cv2.VideoCapture(video_dir)

        if not cap.isOpened():
            raise FileNotFoundError(f"Error opening video file {video_dir}!")

        frames = []
        i = 0
        while cap.isOpened():
            i += 1
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            if frame is None:
                break

        self._frames = frames

    def get_frame(self) -> np.ndarray:
        raw_frame = self._frames[self._i]
        self._i += 1
        try
            anchor_point = self.calib_data_manager.get_anchor_point(self.name)
        except NotCalibratedError as error:
            raise NotCalibratedError("Anchor point not set", error)


    # Camera0
        anchor_point0 = np.array(self.conf["anchor_point"][0])
        start_point0 = anchor_point0 - np.array(self.conf["camera_size"][0]) / 2
        end_point0 = anchor_point0 + np.array(self.conf["camera_size"][0]) / 2

        # Camera 1
        anchor_point1 = np.array(self.conf["anchor_point"][1])
        start_point1 = anchor_point1 - np.array(self.conf["camera_size"][0]) / 2
        end_point1 = anchor_point1 + np.array(self.conf["camera_size"][0]) / 2

        # checking for negative values and adjusting the anchor size
        for i, val in enumerate(start_point0):
            if val < 0:
                self.conf["anchor_point"][0] -= val
                return self(image)

        for i, val in enumerate(start_point1):
            if val < 0:
                self.conf["anchor_point"][1] -= val
                return self(image)

        frame0 = image[int(start_point0[1]): int(end_point0[1]), int(start_point0[0]): int(end_point0[0])]
        frame1 = image[int(start_point1[1]): int(end_point1[1]), int(start_point1[0]): int(end_point1[0])]

        # frame0 should be mirror frame
        frame0 = np.fliplr(frame0)
    def end_of_frames(self) -> bool:
        if self._i >= len(self._frames):
            return True
        else:
            return False

    def reset(self):
        self._i = 0
