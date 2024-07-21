from typing import Union
import numpy as np


def generate_objectpoints(rows_inner: int, columns_inner: int, edge_length: Union[int, float]) -> np.ndarray:
    # prepare object points, lower left corner of chessboard will be world coordinate (0, 0, 0)
    assert isinstance(rows_inner, int)
    assert isinstance(columns_inner, int)
    assert isinstance(edge_length, (float, int))

    objpoint_one_image = np.zeros((columns_inner * rows_inner, 3), np.float32)
    objpoint_one_image[:, :2] = np.mgrid[0:rows_inner, 0:columns_inner].T.reshape(-1, 2)
    objpoint_one_image = edge_length * objpoint_one_image

    return objpoint_one_image


class CornerDetectionError(Exception):
    def __init__(self, message, error_code):
        super().__init__(message)  # Initialize the base Exception class with the message
        self.error_code = error_code  # Additional attribute for the custom error code

    def __str__(self):
        return f"{self.args[0]} (Error Code: {self.error_code})"


class CalibrationError(Exception):
    def __init__(self, message, error_code):
        super().__init__(message)  # Initialize the base Exception class with the message
        self.error_code = error_code  # Additional attribute for the custom error code

    def __str__(self):
        return f"{self.args[0]} (Error Code: {self.error_code})"


class NotCalibratedError(Exception):
    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = error_code

    def __str__(self):
        return f"{self.args[0]} (Error Code: {self.error_code})"
