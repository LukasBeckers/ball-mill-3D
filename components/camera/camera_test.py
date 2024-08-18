import numpy as np
import os
from PIL import Image
from numpy.lib import utils

from . import (
    Camera,
    ICalibrationDataManager,
    NotCalibratedError,
    ICornerDetector,
    ICameraFrameProvider)

current_directory = os.path.dirname(os.path.abspath(__file__))

class TestCalibrationManager(ICalibrationDataManager):
    def __init__(self):
        self.camera_matrix = None
        self.distortion_matrix = None
        self.optimized_camera_matrix = None
        self.camera_resolution = None

    def get_camera_matrix(self, name: str) -> np.ndarray:
        if self.camera_matrix is not None:
            return self.camera_matrix
        else:
            raise NotCalibratedError("", NotCalibratedError)

    def get_distortion_matrix(self, name: str) -> np.ndarray:
        if self.distortion_matrix is not None:
            return self.distortion_matrix
        else:
            raise NotCalibratedError("", NotCalibratedError)

    def get_optimized_camera_matrix(self, name: str) -> np.ndarray:
        if self.optimized_camera_matrix is not None:
            return self.optimized_camera_matrix
        else:
            raise NotCalibratedError("", NotCalibratedError)

    def get_camera_resolution(self, name: str) -> np.ndarray:
        if self.camera_resolution is not None:
            return self.camera_resolution
        else:
            raise NotCalibratedError("", NotCalibratedError)

    def save_camera_matrix(self, name: str, camera_matrix: np.ndarray):
        print("Saving Camera Marix:\n", camera_matrix, "Name", name )
        self.camera_matrix = camera_matrix

    def save_calibration_error(self, name: str, calibration_error: np.ndarray):
        print("Saving Calibration Error:\n", calibration_error, "Name", name)
        self.calibration_error = calibration_error

    def save_distortion_matrix(self, name: str, distortion_matrix: np.ndarray):
        print("Saving Distortion Matrix:\n", distortion_matrix, "Name", name)
        self.distortion_matrix = distortion_matrix

    def save_optimized_camera_matrix(
        self, name: str, optimized_camera_matrix: np.ndarray
    ):
        print("Saving Optimized Camera Matrix:\n", optimized_camera_matrix, "Name", name)
        self.optimized_camera_matrix = optimized_camera_matrix

    def save_camera_resolution(self, name: str, camera_resolution: np.ndarray):
        print("Saving Camera Resolution:\n", camera_resolution, "Name", name)
        self.camera_resolution = camera_resolution


class TestFrameProvider(ICameraFrameProvider):
    def __init__(self):
        self.i = 0
        self.frames = [np.array(Image.open(os.path.join(current_directory,"test_data/images/test1.jpg")))]

    def get_frame(self) -> np.ndarray:
        self.i += 1
        return self.frames[self.i - 1]


    def end_of_frames(self) -> bool:
        if self.i >= len(self.frames):
            return True
        else:
            return False

    def reset(self):
        self.i = 0


class TestCornerDetector(ICornerDetector):
    def detect_corners(
        self, image: np.ndarray, rows_inner: int, columns_inner: int
    ) -> np.ndarray:
        print("Returning corner_detection")
        return np.array([[271.52875, 239.39052],
         [271.03214, 180.79181],
         [270.51706, 120.01276],
         [336.18893, 240.30408],
         [336.93515, 181.60043],
         [337.7117,  120.51197],
         [398.99,    241.19138],
         [401.7363,  182.39554],
         [404.6036,  121.00894],
         [464.03824, 242.11043],
         [467.26965, 183.19963],
         [470.65402, 121.49966]])


def test_camera():
    test_calibration_manager = TestCalibrationManager()
    test_calibration_manager.save_camera_resolution("test", np.array((360, 180)))
    print("Testing camera initialization...", end="")
    test_cam = Camera(name = "test", calibration_manager=test_calibration_manager)
    print("   OK!")
    test_frame_provider = TestFrameProvider()
    test_corner_detector = TestCornerDetector()
    print("Test Calibration ...")
    test_cam.calibrate(
        frame_providers=[test_frame_provider],
        corner_detector=test_corner_detector,
        rows_inner=3,
        columns_inner=4,
        edge_length=0.005
    )
    print("Calibration   OK!")
