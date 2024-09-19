from abc import ABC, abstractmethod
from typing import Union, List, Tuple
import cv2
import numpy as np
from .camera import (
    Camera,
    ICameraFrameProvider,
    ICornerDetector,
    CornerDetectionError,
    generate_objectpoints,
)

from .camera_utils import (
    CornersOrdererError,
    ImagePointsExtractionError,
    StereoCalibrationError,
)


class IStereoCornerOrderManager(ABC):
    @abstractmethod
    def oder_corners(
        self,
        corners0: np.ndarray,
        corners1: np.ndarray,
        frame0: np.ndarray,
        frame1: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


class IStereoCalibrationDataManager(ABC):

    @abstractmethod
    def get_translation_matrix(self, name: str) -> np.ndarray:
        pass

    @abstractmethod
    def get_rotation_matrix(self, name: str) -> np.ndarray:
        pass

    @abstractmethod
    def save_translation_matrix(
        self, name: str, translation_matrix: np.ndarray
    ):
        pass

    @abstractmethod
    def save_rotation_matrix(self, name: str, rotation_matrix: np.ndarray):
        pass

    @abstractmethod
    def save_stereo_calibration_error(
        self, name: str, stereo_calibration_error: np.ndarray
    ):
        pass


class StereoCamera:
    def __init__(
        self,
        name: str,
        camera0: Camera,
        camera1: Camera,
        corner_order_manager: IStereoCornerOrderManager,
        stereo_calibration_manager: IStereoCalibrationDataManager,
        corner_detector: ICornerDetector,
    ):
        self.name = name
        self.camera0 = camera0
        self.camera1 = camera1
        self.corner_order_manger = corner_order_manager
        self.stereo_calibration_manager = stereo_calibration_manager
        self.corner_detector = corner_detector

    def get_translation_matrix(self) -> np.ndarray:
        return self.stereo_calibration_manager.get_translation_matrix(
            self.name
        )

    def get_rotation_matrix(self) -> np.ndarray:
        return self.stereo_calibration_manager.get_rotation_matrix(self.name)

    def save_translation_matrix(self, translation_matrix: np.ndarray):
        self.stereo_calibration_manager.save_translation_matrix(
            self.name, translation_matrix
        )

    def save_rotation_matrix(self, rotation_matrix: np.ndarray):
        self.stereo_calibration_manager.save_rotation_matrix(
            self.name, rotation_matrix
        )

    def save_stereo_calibration_error(
        self, stereo_calibration_error: np.ndarray
    ):
        self.stereo_calibration_manager.save_stereo_calibration_error(
            self.name, stereo_calibration_error
        )

    def _detect_stereo_corners(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        rows_inner: int,
        columns_inner: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            corners0 = self.corner_detector.detect_corners(
                image=frame0,
                rows_inner=rows_inner,
                columns_inner=columns_inner,
            )
        except CornerDetectionError as e:
            raise CornerDetectionError("Corner detection failed", e)

        try:
            corners1 = self.corner_detector.detect_corners(
                image=frame1,
                rows_inner=rows_inner,
                columns_inner=columns_inner,
            )
        except CornerDetectionError as e:
            raise CornerDetectionError("Corner detection failed", e)
        return corners0, corners1

    def _get_imagepoints(
        self,
        frame_provider0,
        frame_provider1,
        rows_inner: int,
        columns_inner: int,
    ):

        frame0 = frame_provider0.get_stereo_frames()
        frame1 = frame_provider1.get_stereo_frames()

        try:
            corners0, corners1 = self._detect_stereo_corners(
                frame0, frame1, rows_inner, columns_inner
            )
        except CornerDetectionError as e:
            raise ImagePointsExtractionError("Corner detection failed", e)

        try:
            corners0, corners1 = self.corner_order_manger.oder_corners(
                corners0, corners1, frame0, frame1
            )
        except CornersOrdererError as e:
            raise ImagePointsExtractionError("Corners ordering failed", e)

        imagepoint0 = corners0
        imagepoint1 = corners1

        return imagepoint0, imagepoint1

    def stereo_calibrate(
        self,
        frame_provider0: ICameraFrameProvider,
        frame_provider1: ICameraFrameProvider,
        rows_inner: int = 7,
        columns_inner: int = 9,
        edge_length: float = 0.005,
        undistort: bool = False,
        stereocalibration_flags=cv2.CALIB_FIX_PRINCIPAL_POINT,
    ):
        assert isinstance(
            frame_provider0, ICameraFrameProvider
        ), "frame_provider0 should be of type ICameraFrameProvider"
        assert isinstance(
            frame_provider1, ICameraFrameProvider
        ), "frame_provider1 should be of type ICameraFrameProvider"
        assert isinstance(rows_inner, int), "rows should be of type int"
        assert isinstance(columns_inner, int), "columns should be of type int"
        assert isinstance(
            edge_length, float
        ), "edge_length should be of type float"
        assert isinstance(undistort, bool), "undistort should be of type bool"

        imgpoints0 = []  # Pixel Coordinates
        imgpoints1 = []
        objpoints = []  # World Coordinates
        objpoint_one_image = generate_objectpoints(
            rows_inner=rows_inner,
            columns_inner=columns_inner,
            edge_length=edge_length,
        )

        while (
            not frame_provider0.end_of_frames()
            and not frame_provider1.end_of_frames()
        ):
            try:
                imagepoint0, imagepoint1 = self._get_imagepoints(
                    frame_provider0, frame_provider1, rows_inner, columns_inner
                )
            except ImagePointsExtractionError as e:
                print("Image point extraction failed")
                continue

            imgpoints0.append(imagepoint0)
            imgpoints1.append(imagepoint1)
            objpoints.append(objpoint_one_image)

        if objpoints == [] or imgpoints0 == [] or imgpoints1:
            raise StereoCalibrationError(
                """Stereo calibration failed, 
                no imagepoints were extracted, 
                or no object_points were generated""",
                StereoCalibrationError,
            )

        frame_provider0.reset()
        example_frame = frame_provider0.get_frame()
        height, width = example_frame.shape[:2]
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            1000,
            0.00001,
        )
        (
            stereo_calibration_error,
            camera_matrix0,
            distortion_matrix0,
            camera_matrix1,
            distortion_matrix1,
            R,
            T,
            E,
            F,
        ) = cv2.stereoCalibrate(
            objpoints,
            imgpoints0,
            imgpoints1,
            self.camera0.get_optimized_camera_matrix(),
            self.camera0.get_distortion_matrix(),
            self.camera1.get_optimized_camera_matrix(),
            self.camera1.get_distortion_matrix(),
            (width, height),
            criteria=criteria,
            flags=stereocalibration_flags,
        )

        optimized_camera_matrix0, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix0,
            distortion_matrix0,
            (width, height),
            1,
            (width, height),
        )
        optimized_camera_matrix1, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix1,
            distortion_matrix1,
            (width, height),
            1,
            (width, height),
        )
        print(f"Stereo-calibration error: {stereo_calibration_error}")
        print(f"Translation Matrix: {T}")
        print(f"Rotation Matrix: {R}")

        self.save_stereo_calibration_error(stereo_calibration_error)
        self.save_translation_matrix(T)
        self.save_rotation_matrix(R)
