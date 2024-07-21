import numpy as np
import cv2


def triangulate(SC, point1, point2):
    RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    RT2 = np.concatenate([SC.conf["rotation_matrix"][0], SC.conf["translation_matrix"][0]], axis=-1)

    P1 = SC.conf["camera_matrix"][0] @ RT1
    P2 = SC.conf["camera_matrix"][1] @ RT2

    cm0 = np.array(SC.conf["camera_matrix"][0])
    cm1 = np.array(SC.conf["camera_matrix"][1])
    dist0 = np.array(SC.conf["distortion"][0])
    dist1 = np.array(SC.conf["distortion"][0])
 
    point1 = cv2.undistortPoints(point1, cm0, dist0 , None, cm0)[0][0]
    point2 = cv2.undistortPoints(point2, cm1, dist1, None, cm1)[0][0]

    coordinate = cv2.triangulatePoints(P1, P2, point1, point2)
    coordinate = (coordinate[:3, :] / coordinate[3, :]).T[0]

    return coordinate


def compute_transform_matrix(points_A, points_B):
    """Calculates the transform matrix to transform any point of one coordinatesystem (3d) to another"""
    # Assuming points_A and points_B are numpy arrays with shape (N, 3)
    # where N is the number of points (N >= 3),
    # Reshaping the points
    points_A = np.array(points_A)
    points_B = np.array(points_B)
    points_A = points_A.T
    points_B = points_B.T

    # Calculate the centroids of both point sets
    centroid_A = np.mean(points_A, axis=1, keepdims=True)
    centroid_B = np.mean(points_B, axis=1, keepdims=True)

    # Compute the centered point sets
    centered_A = points_A - centroid_A
    centered_B = points_B - centroid_B

    # Compute the covariance matrix
    covariance_matrix = centered_A @ centered_B.T

    # Perform Singular Value Decomposition
    U, _, Vt = np.linalg.svd(covariance_matrix)

    # Calculate the rotation matrix
    rotation_matrix = Vt.T @ U.T

    # Calculate the translation vector
    translation_vector = centroid_B - rotation_matrix @ centroid_A

    # Create the transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation_vector.flatten()
    return transform_matrix

def transform_point(point, transform_matrix):
    """Transforms a point to a nother coordinate system (3d) based on a transform_matrix"""
    # point needs one extra dimension
    point = np.array([point])
    point_homogeneous = np.hstack((point, np.ones((1, 1))))
    new_point_homogeneous = transform_matrix @ point_homogeneous.T

    # Extract the transformed point in system B
    new_point = new_point_homogeneous[:3, :].T
    return new_point[0]


