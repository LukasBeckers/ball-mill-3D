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
    def set_anchor_point(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def set_is_mirrored(self, image: np.ndarray) -> bool:
        pass


class CameraFrameProvider(ICameraFrameProvider):
    def __init__(
        self,
        camera: Camera,
        frame_provider_calibration_data_manager: IFrameProviderCalibrationDataManager,
        gui: IFrameProviderGUI,
    ):
        self.camera = camera
        self.name = camera.name
        self._i = 0
        self._frames = []
        self.calibration_data_manager = frame_provider_calibration_data_manager
        self.gui = gui

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

    def calibrate(self, video_dir: str):
        self._load_frames(video_dir)
        raw_frame = self._frames[
            int(len(self._frames) / 2)
        ]  # Frame from the middle of the video
        anchor_point = self.gui.set_anchor_point(raw_frame)
        is_mirrored = self.gui.set_is_mirrored(raw_frame)
        self.calibration_data_manager.save_anchor_point(self.name, anchor_point)
        self.calibration_data_manager.save_is_mirrored(self.name, is_mirrored)

    def get_frame(self) -> np.ndarray:
        raw_frame = self._frames[self._i]
        self._i += 1
        try:
            anchor_point = self.calibration_data_manager.get_anchor_point(self.name)
        except NotCalibratedError as error:
            raise NotCalibratedError("Anchor point not set", error)

        try:
            is_mirrored = self.calibration_data_manager.get_is_mirrored(self.name)
        except NotCalibratedError as error:
            raise NotCalibratedError("is_mirrored is not set", error)

        try:
            camera_resolution = self.camera.get_camera_resolution()
        except NotCalibratedError as error:
            raise NotCalibratedError(
                f"Camera {self.camera.name}, camera_resolution is not set", error
            )

        start_point = anchor_point - camera_resolution / 2
        end_point = anchor_point + camera_resolution / 2

        # checking for negative values and adjusting the anchor size
        for i, val in enumerate(start_point):
            if val < 0:
                self.calibration_data_manager.save_anchor_point(
                    self.name, anchor_point - val
                )
                return self.get_frame()

        frame = raw_frame[
            int(start_point[1]) : int(end_point[1]),
            int(start_point[0]) : int(end_point[0]),
        ]

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
