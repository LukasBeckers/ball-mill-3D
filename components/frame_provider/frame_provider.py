from abc import ABC, abstractmethod
import numpy as np
import cv2

from camera import ICameraFrameProvider, Camera, NotCalibratedError


class IFrameProviderCalibrationDataManager(ABC):
    @abstractmethod
    def get_anchor_point(self, name: str) -> np.ndarray:
        pass

    @abstractmethod
    def get_is_mirrored(self, name: str) -> bool:
        pass

    @abstractmethod
    def save_anchor_point(self, name: str, anchor_point: np.ndarray):
        pass

    @abstractmethod
    def save_is_mirrored(self, name: str, is_mirrored: bool):
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
        self.name = camera.name
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
        try:
            anchor_point = self.calib_data_manager.get_anchor_point(self.name)
        except NotCalibratedError as error:
            raise NotCalibratedError("Anchor point not set", error)

        try:
            is_mirrored = self.calib_data_manager.get_is_mirrored(self.name)
        except NotCalibratedError as error:
            raise NotCalibratedError("is_mirrored is not set", error)

        try:
           camera_resolution = self.camera.get_camera_resolution()
        except NotCalibratedError as error:
            raise NotCalibratedError(f"Camera {self.camera.name}, camera_resolution is not set", error)

        start_point = anchor_point - camera_resolution / 2
        end_point = anchor_point + camera_resolution / 2

        # checking for negative values and adjusting the anchor size
        for i, val in enumerate(start_point):
            if val < 0:
                self.calib_data_manager.save_anchor_point(self.name, anchor_point-val)
                return self.get_frame()


        frame = raw_frame[int(start_point[1]): int(end_point[1]), int(start_point[0]): int(end_point[0])]

        if is_mirrored:
            frame = np.fliplr(frame)

        return frame

    def end_of_frames(self) -> bool:
        if self._i >= len(self._frames):
            return True
        else:
            return False

    def reset(self):
        self._i = 0
