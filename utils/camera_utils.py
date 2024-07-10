"""
What I painfully learned in this file:
    When working with open-cv always copy an object before passing it to any function.
    Open-cv tends to change input instances!

"""

import cv2
import torch
import uuid
import pickle as pk
from collections import defaultdict
import yaml
from yaml import SafeLoader
import os

from utils.video_utils import *

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))


def switch_rows(corners, len_row):
    "Switches the rows in the chessboard corner detection"
    switched_corners = []
    row = []
    for corner in corners:
        row.insert(0, corner)
        if len(row) == len_row:
            switched_corners.extend(row)
            row = []
    return np.array(switched_corners)


def draw_lines(img):
    global line  # line = (x1, y1, x2, y2)
    line = []
    lines = []
    def mouse_callback(event, x, y, flags, param):
        global line
        if event == cv2.EVENT_LBUTTONDOWN:
            line.extend((x, y))

    img_old = np.array(img)
    img = np.array(img)
    while True:
        cv2.putText(img, f'Left Click: line point, R: undo line, space: finish', (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, f'Press "d" to skipp this image!', (10, 51),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Drawing Lines", img)
        cv2.setMouseCallback("Drawing Lines", mouse_callback)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('r'):
            if len(lines) > 0:
                lines.pop()
            img = np.array(img_old)
            for l in lines:
                cv2.line(img, (l[0], l[1]), (l[2], l[3]), color=(0, 255, 0), thickness=1)
        if key & 0xFF == 32:
            cv2.destroyWindow("Drawing Lines")
            break
        if key & 0xFF == ord("d"): # Pressing d skipps this image
            cv2.destroyWindow("Drawing Lines")
            return None, None
        if len(line) == 4:
            lines.append(line)
            line = []
            img = np.array(img_old)
            for l in lines:
                cv2.line(img, (l[0], l[1]), (l[2], l[3]), color=(0, 255, 0), thickness=1)
    return lines, img_old


def line_intersection(lines):
    """
    Finds the intersection of two lines given in Hesse normal form.
    """
    intersections = []
    for i, line0 in enumerate(lines):
        for line1 in lines[i + 1:]:
            x1, y1, x2, y2 = line0
            x3, y3, x4, y4 = line1
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + 1e-10)
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + 1e-10)
            intersections.append((px, py))
    return np.array(intersections)


def label_corners(img, image_scaling=2):
    """
    Allows the user to draw lines on an image, the line intersection will be interpreted as
    corners in a chessboard pattern.
    """
    img = np.array(img)
    try:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    except Exception:
        pass
    
    # Scaling the image for better visibility
    img = cv2.resize(img, [int(img.shape[1]*image_scaling), int(img.shape[0]*image_scaling)])
    # Draws the lines (by user)
    lines, img = draw_lines(img)
    # The user can decide in draw_lines to mark this image for skipping then draw_lines returns None, None
    if img is None:
        return None
    # Calculates the corners via line-intersection
    corners = line_intersection(lines)
    w, h = img.shape[:2]
    # Drops corners that are outside the image
    corners = np.array([[corner] for corner in corners if h > corner[0] > 0 and w > corner[1] > 0], dtype=np.float32)
    # Refining the corner detections
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00001)
    # Saves the corners and the image for later usage in a corner-detection project.
    if len(corners) > 0:
        os.makedirs(f'{current_dir}/../labeled_chess_boards/images', exist_ok=True)
        os.makedirs(f'{current_dir}/../labeled_chess_boards/corners', exist_ok=True)
        id = uuid.uuid4()
        cv2.imwrite(f'{current_dir}/../labeled_chess_boards/images/{id}.jpg',
                img)
        np.save(f'{current_dir}/../labeled_chess_boards/corners/{id}.npy', corners)

    # Reversing the Scaling
    corners /= image_scaling 
    return corners


def draw_cutout_corners(img, cam="no cam specified"):
    """
    Left-click on the image to mark a point, two points will span a rectangle, which will be used
    as a stencil to create cutouts from the image.

    You can draw as many points as you like, only the last two inputs will be used.
    
    You can also press "d" to return None (which can be used as an indicator to skip this image)
    """
    global cutout_corners
    cutout_corners = []
    def mouse_callback(event, x, y, flags, param):
        global cutout_corners
        if event == cv2.EVENT_LBUTTONDOWN:
            cutout_corners.append((x, y))
    
    while True:        
        img_show = np.array(img)
        for point in cutout_corners[-2:]:
            cv2.circle(img_show, point, 2, (0, 255, 0), 2)
        cv2.putText(img_show,
                    f'Click on the corners of the chess board to cut out the image section',
                    (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 255, 0),
                    1)
        cv2.putText(img_show,
                    f'Press "d" to skip this image!',
                    (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 255, 0),
                    1)
        cv2.imshow(f'Camera: {cam}', img_show)
        cv2.setMouseCallback(f"Camera: {cam}", mouse_callback)
        key = cv2.waitKey(1)
        if key & 0xFF == 32: # Press space to quit after drawing the points
            if len(cutout_corners) >= 2:
                cv2.destroyWindow(f"Camera: {cam}")
                break
        if key & 0xFF == ord("d"): # Press d to label this image for skipping.
                cv2.destroyWindow(f"Camera: {cam}")
                return None
                break
    return np.array(cutout_corners[-2:], dtype=np.int32)


def show_and_switch(img0, img1, corners0, corners1, rows_inner, columns_inner, image_scaling):
    img0_old = np.array(img0)
    img1_old = np.array(img1)
    img0 = cv2.resize(img0, (np.array(img0.shape[:2])[::-1] * image_scaling).astype(np.int32))
    img1 = cv2.resize(img1, (np.array(img1.shape[:2])[::-1] * image_scaling).astype(np.int32))
    corners0 *= image_scaling
    corners1 *= image_scaling
    
    cv2.putText(img0, f's: switch order of all points',
                (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img0, f'l = switch lines',
                (10, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img0, f'd = delete detection',
                (10, 90), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    try:
        cv2.drawChessboardCorners(img0, (rows_inner, columns_inner), corners0, True)
    except Exception:
        pass
    for i, [corner] in enumerate(corners0):
        cv2.putText(img0, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (0, 255, 0), 1)
    try:
        cv2.drawChessboardCorners(img1, (rows_inner, columns_inner), corners1, True)
    except Exception:
        pass
    for i, [corner] in enumerate(corners1):
        cv2.putText(img1, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (0, 255, 0), 1)
    img0_name = 'Show and Switch Detection 1'
    img1_name = 'Show and Switch Detection 2'
    cv2.imshow(img0_name, img0)
    cv2.imshow(img1_name, img1)
    key = cv2.waitKey(0)

    if key & 0xFF == ord('s'):  # press s to switch ordering of img0
        cv2.destroyWindow(img0_name)
        corners0 = corners0[::-1]
        return show_and_switch(np.array(img0_old), np.array(img1_old), corners0 / image_scaling, corners1 / image_scaling, rows_inner, columns_inner, image_scaling)

    if key & 0xFF == ord("l"):  # press l to switch lines in img0
        cv2.destroyWindow(img0_name)
        corners0 = switch_rows(corners0, rows_inner)
        img0 = np.array(img0_old)
        return show_and_switch(np.array(img0_old), np.array(img1_old), corners0 / image_scaling, corners1 / image_scaling, rows_inner, columns_inner, image_scaling)
    
    if key & 0xFF == ord("d"): # Deleting this detection
        cv2.destroyWindow(img0_name)
        cv2.destroyWindow(img1_name)
        return None
    
    if key & 0xFF == 32: # press space to finish
        cv2.destroyWindow(img0_name)
        cv2.destroyWindow(img1_name)
        return corners0 / image_scaling, corners1 / image_scaling

    # Nothing happenes if another key is pressed
    return show_and_switch(np.array(img0_old), np.array(img1_old), corners0 / image_scaling, corners1 / image_scaling, rows_inner, columns_inner, image_scaling)


def print_corners(image, corners, optimized, rows_inner, columns_inner, image_scaling):
    # Shows one example of the drawn chessboard corners
    cv2.drawChessboardCorners(image,
                            (rows_inner, columns_inner),
                            corners * image_scaling,
                            True)
                                 
    for i, [corner] in enumerate(corners):
        cv2.putText(image,
                    f'{i}',
                    (int(corner[0] * image_scaling),
                    int(corner[1] * image_scaling)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 255, 0),
                    1)
    cv2.putText(image, "Space: accept, d: decline", 
        (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0),  1)
    cv2.putText(image, "o: toggle optimizaiton", 
        (10, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0),  1)
    if optimized:
        cv2.putText(image, "OPTIMIZED CORNERS",
            (10, 90), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0),  1)
    else:
        cv2.putText(image, "Un-OPTIMIZED CORNERS",
            (10, 90), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0),  1)
    return image


def corners_sanity_checker(corners, rows_inner, columns_inner, example_image):
    n_corners = rows_inner * columns_inner

    if corners is None:
        return False
    if len(corners) == n_corners:
        pass 
    else: 
        return False
    for corner in corners: # Checking if corner is in image
        x, y = corner.ravel()
        if x < 0 or x >= example_image.shape[1] or y < 0 or y >= example_image.shape[0]:
            return False
    return True


def corner_detection(image_set, image_scaling, cam, rows_inner, columns_inner, fallback_manual):
    image_set = [cv2.resize(img, (np.array(img.shape[:2])[::-1] * image_scaling).astype(int)) for img in image_set]
    image_set_old = np.array(image_set)
    man_image_set_old = image_set_old[:1]
    
    # Drawing the cutout_corners
    cc = draw_cutout_corners(image_set[0], cam)

    if cc is None:
        return None

    offset = cc.min(axis=0)
    image_set = [img[cc[:, 1].min():cc[:, 1].max(), cc[:, 0].min():cc[:, 0].max()] for img in image_set] # Generate the coutouts
    image_set = [cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) for img in image_set] # Convert to gray
    man_image_set = image_set[:1] # For manual labeling only one image is used
    
    corners_set = [cv2.findChessboardCorners(img, (rows_inner, columns_inner), None) for img in image_set]
    # Removes the image_set of unsuccessful corner-predictions. res[1] = corner-coordinates, res[1] = ret
    image_set = [image_set[i] for i, res in enumerate(corners_set) if res[0]]
    image_set_old = [image_set_old[i] for i, res in enumerate(corners_set) if res[0]]
    corners_set = [res[1] for res in corners_set if res[0]] # Removes unsuccessful detections. res[1] = corner-coordinates, res[0] = ret
    while True: # Handeling fallback to manual prediction in a loop
        detection_success = corners_sanity_checker(corners_set[0] if corners_set != [] else None, rows_inner, columns_inner, man_image_set[0])

        if detection_success: # Success -> loop breaks
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            optimized_corners_set = [cv2.cornerSubPix(np.copy(img), np.copy(image_corners), (11, 11), (-1, -1), criteria)
                for img, image_corners in zip(image_set, corners_set)]
            break

        elif not detection_success and fallback_manual: 
            image_set = man_image_set
            image_set_old = man_image_set_old
            corners_set = [label_corners(image_set_old[0], image_scaling=1)]
            corners_set = [np.array([np.array(coord) - offset for coord in corners_set[0]], dtype=np.float32) if corners_set[0] is not None else None] # Removes the offset to make corners identical to automatically detected corners
            fallback_manual = False
            continue

        elif not detection_success and not fallback_manual:
            return None # detection failed
   
    corners_set = [np.array([(np.array(coord) + offset) / image_scaling for (coord) in image_corners], dtype=np.float32)
            for image_corners in corners_set] # Reversing the image scaling
    optimized_corners_set = [np.array([(np.array(coord) + offset) / image_scaling for (coord) in image_corners], dtype=np.float32)
            for image_corners in optimized_corners_set] # Reversing the image scaling
    
    optimized = True # Toggel user can switch to False if the optimizations are not good
    while True:
        img_show = print_corners(np.array(image_set_old[0]), 
                                 optimized_corners_set[0] if optimized else corners_set[0],
                                 optimized,
                                 rows_inner, 
                                 columns_inner,
                                 image_scaling) 
        cv2.imshow(f'Camera: {cam}', img_show)
        key = cv2.waitKey(0)
        if key & 0xFF == ord("d"): # Deleting this detection           
            cv2.destroyWindow(f'Camera: {cam}')
            delete = True
            break

        if key & 0xFF == 32: # press space to finish
            cv2.destroyWindow(f'Camera: {cam}')
            delete = False
            break 

        if key & 0xFF == ord("o"): # press o to switch between optimized and unoptimized corners_set 
            cv2.destroyWindow(f'Camera: {cam}')
            optimized = not optimized

    if delete:
        return None
    if optimized:
        corners_set = optimized_corners_set

    return corners_set


def generate_objectpoints(rows_inner, columns_inner, edge_length):

    # prepare object points, lower left corner of chessboard will be world coordinate (0, 0, 0)
    objpoint_one_image = np.zeros((columns_inner * rows_inner, 3), np.float32)
    objpoint_one_image[:, :2] = np.mgrid[0:rows_inner, 0:columns_inner].T.reshape(-1, 2)
    objpoint_one_image = edge_length * objpoint_one_image
    return objpoint_one_image


class stereoCamera():
    def __init__(self, name="noName", **kwargs):
        """
        Possible **kwargs are all dicts with the keys = int(camera_index) ie. 0 or 1:

        camera_size e.g. camera_size={0: [100, 100], 1:[100,100]}
        other possible / meaningfull kwargs_dicts are:

            anchor_point
            projection_error
            camera_matrix
            optimized_camera_matrix 
            distortion
        """
        def recursive_defaultdict():
            return defaultdict(lambda: None)

        self.conf = defaultdict(recursive_defaultdict,
                                {key: defaultdict(lambda: None, {k: v for k, v in value.items()})
                                 for key, value in kwargs.items()})
        self.name = name

    def save_to_yaml(self, filename="camera_config.yaml"):
        """
        Saves the calculated parameters like "camera_matrix" etc., stored in self.conf to a yaml file
        the yaml file can be found at BallMill3d/config/<config_name>.yaml
        """
        def convet_to_list(obj):
            # converts the default dict to a list.
            if isinstance(obj, (list, tuple)):
                return [convet_to_list(o) for o in obj]
            elif isinstance(obj, (np.ndarray, torch.Tensor)):
                return obj.tolist()
            return obj
        data_to_save = {'name': self.name}
        for key, value in self.conf.items():
            value = {k: convet_to_list(v) for k, v in value.items()}
            data_to_save[key] = value
        # Writing to a yaml file
        filename = f"config/{filename}"
        with open(os.path.join(os.path.dirname(current_dir), filename), 'w') as file:
            yaml.dump(data_to_save, file)

    def load_from_yaml(self, filename=f"camera_config.yaml"):
        """
        Loads the configs stored to a yaml file by "save_to_yaml" method
        """
        filename = f"config/{filename}"
        with open(os.path.join(os.path.dirname(current_dir), filename), 'r') as file:
            data_loaded = yaml.load(file, Loader=SafeLoader)
        # Reinitializing the instance variables
        self.name = data_loaded.pop('name')

        def recursive_defaultdict():
            return defaultdict(lambda: None)

        self.conf = defaultdict(recursive_defaultdict,
                                {key: defaultdict(lambda: None, {k: v for k, v in value.items()})
                                 for key, value in data_loaded.items()})

    def undistort_image(self, img, cam):
        """
        Un-distorts an image using the camera_matrix and the distortion values obtained by camera calibration.
        """

        optim_camera_matrix = self.conf["optimized_camera_matrix"][cam]
        img = cv2.undistort(img, np.array(self.conf["camera_matrix"][cam]),
                            np.array(self.conf["distortion"][cam]),
                            None,
                            np.array(optim_camera_matrix))
        return img


    def calibrate(self,
                  image_sets:list,
                  cam:int, 
                  rows:int=8,
                  columns:int=10,
                  edge_length:float=0.005,
                  image_scaling:float=2,
                  fallback_manual:bool=False,
                  optimize_manual_predictions:bool=False
                  ):
        """
        Calibrates a single camera of this stereoCamera instance.
        rows and columns need to be the real number of rows and columns in the chessboard-pattern

        image_sets are lists of lists of images, use one list of image for each video
        """
        # Only chessboard corners with all four sides being squares can be detected. (B W) Therefore the detectable
        # chessboard rows and columns are one less.                                  (W B)
        rows_inner = rows - 1
        columns_inner = columns - 1
        imgpoints = [] # Pixel coorinates in the image
        objpoints = [] # Defined coordinates in real space

        objpoint_one_image = generate_objectpoints(rows_inner=rows_inner,
                                     columns_inner=columns_inner,
                                     edge_length=edge_length)
        for image_set in image_sets:
            image_set = [self(img)[cam] for img in image_set]

            corners = corner_detection(image_set=image_set, 
                                       image_scaling=image_scaling, 
                                       cam=cam,
                                       rows_inner=rows_inner,
                                       columns_inner=columns_inner,
                                       fallback_manual=fallback_manual)

            if corners is None: # Corners can be set to none if detection failed 
                continue 

            imgpoints.extend(corners)
            objpoints.extend([objpoint_one_image for _ in corners]) # because corners were created by a set of similar images
            
        width = image_sets[0][0].shape[1]
        height = image_sets[0][0].shape[0]
        assert imgpoints != [], "No Corners detected!"
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(np.array(objpoints),
                                                           np.array(imgpoints),
                                                           (width, height),
                                                           None,
                                                           None)
        optimized_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1,
                                                                   (width, height))
        print('rmse:', ret)
        print('camera matrix:\n', mtx)
        print('optimized camera matrix:\n', optimized_camera_matrix)
        print('distortion coeffs:\n', dist)
        self.conf["projection_error"][cam] = ret
        self.conf["camera_matrix"][cam] = mtx
        self.conf["optimized_camera_matrix"][cam] = optimized_camera_matrix
        self.conf["distortion"][cam] = dist
        cv2.destroyAllWindows()
        return

    def stereo_calibrate(self,
                         image_sets,
                         rows=8,
                         columns=10,
                         edge_length=0.005,
                         undistort=True,
                         stereocalibration_flags = cv2.CALIB_FIX_PRINCIPAL_POINT,
                         opip1ip2=None,
                         image_scaling=2,
                         fallback_manual=True):
        
        assert self.conf["camera_matrix"][0] is not None and self.conf["camera_matrix"][1] is not None, \
            "Calibrate both cameras first!"
        assert image_sets[0][0].shape == image_sets[0][1].shape, \
                "Both cameras must have the same resolution for stereo-calibration"

        rows_inner = rows - 1  # only inner chessboard-edges are used         
        columns_inner = columns - 1
        imgpoints_0 = [] # Pixel Coordinates
        imgpoints_1 = []
        objpoints = [] # World Coordinates
        objpoint_one_image = generate_objectpoints(rows_inner=rows_inner, columns_inner=columns_inner, edge_length=edge_length)         

        for image_set in image_sets:
            
            if opip1ip2 is not None: # opip1ip2 can be used to pass objpoints and imagepoints from previous detections (For Debugging).
                continue
            
            image_set_0 = []
            image_set_1 = []
            for image in image_set: # Cutout the camera-regions from the images
                img_0, img_1 = self(image)
                image_set_0.append(img_0)
                image_set_1.append(img_1) 


            if undistort:
                image_set_0 = [self.undistort_image(img, 0) for img in image_set_0]
                image_set_1 = [self.undistort_image(img, 1) for img in image_set_1]

            corners_0 = corner_detection(image_set=image_set_0,
                                         image_scaling=image_scaling,
                                         cam=0, 
                                         rows_inner=rows_inner,
                                         columns_inner=columns_inner,
                                         fallback_manual=fallback_manual)
                                        
            if corners_0 is None:
                continue
            
            corners_1 = corner_detection(image_set=image_set_1,
                                         image_scaling=image_scaling,
                                         cam=1, 
                                         rows_inner=rows_inner,
                                         columns_inner=columns_inner,
                                         fallback_manual=fallback_manual)

            if corners_1 is None:
                continue

            else:
                corners_0, corners_1 = show_and_switch(img0=image_set_0[0],
                                                       img1=image_set_1[0], 
                                                       corners0=corners_0[0], 
                                                       corners1=corners_1[0], 
                                                       rows_inner=rows_inner, 
                                                       columns_inner=columns_inner, 
                                                       image_scaling=image_scaling)
                imgpoints_0.extend([corners_0])
                imgpoints_1.extend([corners_1])
                objpoints.extend([objpoint_one_image]) # In show and switch each corner_set is reduced to only the first image_corners. This can be changed in the future to increase precision
                # append could be used here instead of extend.
        
        height, width = image_sets[0][1].shape[:2]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00001)
        assert objpoints != [], "No Corners were detected!"
        ret, CM0, dist0, CM1, dist1, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                                      imgpoints_0,
                                                                      imgpoints_1,
                                                                      np.array(self.conf["optimized_camera_matrix"][0])  if undistort else np.array(self.conf["camera_matrix"][0]),
                                                                      np.array(self.conf["distortion"][0]),
                                                                      np.array(self.conf["camera_matrix"][1]) if undistort else np.array(self.conf["camera_matrix"][1]),
                                                                      np.array(self.conf["distortion"][1]),
                                                                      (width, height),
                                                                      criteria=criteria,
                                                                      flags=stereocalibration_flags)

        optimized_camera_matrix0, roi = cv2.getOptimalNewCameraMatrix(CM0, dist0, (width, height), 1,
                                                                   (width, height))
        optimized_camera_matrix1, roi = cv2.getOptimalNewCameraMatrix(CM1, dist1, (width, height), 1,
                                                                   (width, height))
        # Matrix that rotates the coordinate system of the second camera to match the first.
        self.conf["rotation_matrix"][0] = R
        # Matrix that translates the coordinate system of the second camera to match the first.
        self.conf["translation_matrix"][0] = T
        self.conf["stereo_calibration_error"][0] = ret
        self.conf["camera_matrix"][0] = CM0
        self.conf["camera_matrix"][1] = CM1
        self.conf["optimized_camera_matrix"][0] = optimized_camera_matrix0
        self.conf["optimized_camera_matrix"][1] = optimized_camera_matrix1
        self.conf["distortion"][0] = dist0
        self.conf["distortion"][1] = dist1
        print(f'Stereo-calibration error: {ret}')
        print(f'Translation Matrix: {T}')
        print(f'Rotation Matrix: {R}')
        print(f"New Camera Matrix Mirror: {CM0}")
        print(f"New Distortion Front", dist1)
        cv2.destroyAllWindows()
        return objpoints, imgpoints_0, imgpoints_1, image_sets

    def set_anchor_point(self, img, cam, image_scaling=1):
        """
        Set an anchor point on the image using the mouse click.
        """
        anchor_point = None  

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                nonlocal anchor_point
                anchor_point = (x, y)

        while True:
            win_name = f"Set Anchor Point - Camera {cam}"
            line0 = f"Left-click where the {'mirror' if cam == 0 else 'frontal'}-camera should point."
            line1 = "Press space when finished!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (0, 255, 0)
            thickness = 2
            img_with_text = img.copy()
            img_with_text = cv2.resize(img_with_text, (np.array(img.shape[:2])[::-1] * image_scaling).astype(int))
            cv2.circle(img_with_text, anchor_point, 2, (0, 255, 0), 2)
            cv2.putText(img_with_text, line0, (10, 30), font, 
                font_scale, color, thickness)
            cv2.putText(img_with_text, line1, (10, 60), font, 
                font_scale, color, thickness)
            cv2.imshow(win_name, img_with_text)
            cv2.setMouseCallback(win_name, mouse_callback)
            key = cv2.waitKey(1)  # Wait indefinitely until a key is pressed
            if key & 0xFF == 32:
                cv2.destroyWindow(win_name)
                break
        anchor_point = np.array(anchor_point)/image_scaling
        if anchor_point is not None:
            self.conf["anchor_point"][cam] = anchor_point

    def draw_camera_region(self, img):
        """
        Draws the camera regions for inspection based on the anchor points and the camera-size
        """
        for anchor_point in self.conf["anchor_point"].values():
            # Drawing camera
            start_point = anchor_point - np.array(self.conf["camera_size"][0])/2
            end_point = anchor_point + np.array(self.conf["camera_size"][0])/2
            img = cv2.rectangle(img,  start_point.astype(np.int32), end_point.astype(np.int32), (255, 0, 0), 5)

        return img

    def __call__(self, image):

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

        # frame0 should be mirror frame, please use this convention
        frame0 = np.fliplr(frame0)

        return np.ascontiguousarray(frame0), np.ascontiguousarray(frame1) 

if __name__ == "__main__":
    pass    


