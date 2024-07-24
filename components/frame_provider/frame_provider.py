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
    def __init__(self, camera_name: str, frame_provider_calibration_data_manager: IFrameProviderCalibrationDataManager, gui_interface: IFrameProviderGUI):
        self.name = camera_name
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



    def end_of_frames(self) -> bool:
        if self._i >= len(self._frames):
            return True
        else:
            return False

    def reset(self):
        self._i = 0
