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
            print("Current line:", line)
        if event == cv2.EVENT_RBUTTONDOWN:
            line.pop()
            line.pop()
            print("Removing last point. Line:", line)

    img_old = np.array(img)
    while True:
        cv2.putText(img, f'Left Click: line point, right: undo point, R: undo line, space: finish', (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.imshow("Drawing Lines", img)
        cv2.setMouseCallback("Drawing Lines", mouse_callback)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('r'):
            print("Removing last line")
            lines.pop()
            img = np.array(img_old)
            for l in lines:
                cv2.line(img, (l[0], l[1]), (l[2], l[3]), color=(0, 255, 0), thickness=1)
        if key & 0xFF == 32:
            # print("Escaping Line drawing")
            cv2.destroyWindow("Drawing Lines")
            break
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
    for i, line1 in enumerate(lines):
        for line2 in lines[i + 1:]:
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + 1e-10)
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + 1e-10)
            intersections.append((px, py))
    return np.array(intersections)


def label_corners(img):
    """
    Allows the user to draw lines on an image, the line intersection will be interpreted as
    corners in a chessboard pattern.
    """
    # Draws the lines (by user)
    lines, img = draw_lines(img)
    # Calculates the corners via line-intersection
    corners = line_intersection(lines)
    w, h = img.shape[:2]
    # Drops corners that are outside the image
    corners = np.array([[corner] for corner in corners if h > corner[0] > 0 and w > corner[1] > 0], dtype=np.float32)
    print("Corners", corners, corners.shape)
    # Refining the corner detections
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00001)
    corners = cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), corners.reshape(-1, 1, 2), (11, 11), (-1, -1), criteria)
    # Saves the corners and the image for later usage in a corner-detection project.
    os.makedirs(f'{current_dir}/../labeled_chess_boards/images', exist_ok=True)
    os.makedirs(f'{current_dir}/../labeled_chess_boards/corners', exist_ok=True)
    id = uuid.uuid4()
    cv2.imwrite(f'{current_dir}/../labeled_chess_boards/images/{id}.jpg',
                img)
    np.save(f'{current_dir}/../labeled_chess_boards/corners/{id}.npy', corners)
    return corners


def draw_cutout_corners(img, cam):
    """
    Left-click on the image to mark a point, two points will span a rectangle, which will be used
    as a stencil to create cutouts from the image.

    You can draw as many points as you like, only the last two inputs will be used.
    """
    global cutout_corners
    cutout_corners = []
    def mouse_callback(event, x, y, flags, param):
        global cutout_corners
        if event == cv2.EVENT_LBUTTONDOWN:
            cutout_corners.append((x, y))
            print("Left mouse button pressed!", cutout_corners)
    
    while True:        
        img_show = np.array(img)
        for point in cutout_corners[-2:]:
            cv2.circle(img_show, point, 2, (0, 255, 0), 2)
        cv2.putText(img_show,
                    f'Click on the corners of the chess board to cut out the image section',
                    (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (255, 255, 255),
                    1)
        cv2.imshow(f'Camera: {cam}', img_show)
        cv2.setMouseCallback(f"Camera: {cam}", mouse_callback)
        key = cv2.waitKey(1)
        # Press space to quit after drawing the points
        if key & 0xFF == 32:
            cv2.destroyWindow(f"Camera: {cam}")
            break
    return np.array(cutout_corners[-2:], dtype=np.int32)


def show_and_switch(img1, img2, corners1, corners2, rows, columns):
    img1_old = np.array(img1)
    cv2.putText(img1, f's: switch order of all, l = switch lines',
                (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    try:
        cv2.drawChessboardCorners(img1, (rows, columns), corners1, True)
    except Exception:
        pass
    for i, [corner] in enumerate(corners1):
        cv2.putText(img1, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 0, 255), 1)
    try:
        cv2.drawChessboardCorners(img2, (rows, columns), corners2, True)
    except Exception:
        pass
    for i, [corner] in enumerate(corners2):
        cv2.putText(img2, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 0, 255), 1)
    cv2.imshow(f'Detection 1', img1)
    cv2.imshow(f'Detection 2', img2)
    key = cv2.waitKey(0)

    if key & 0xFF == ord('s'):  # press s to switch ordering of img1
        cv2.destroyWindow(f'Detection 1')
        corners1 = corners1[::-1]
        img1 = np.array(img1_old)
        return show_and_switch(img1, img2, corners1, corners2)

    if key & 0xFF == ord("l"):  # press l to switch lines in img1
        cv2.destroyWindow(f'Detection 1')
        corners1 = switch_rows(corners1, rows)
        img1 = np.array(img1_old)
        return show_and_switch(img1, img2, corners1, corners2)

    cv2.destroyWindow(f'Detection 1')
    cv2.destroyWindow(f'Detection 2')
    return corners1, corners2


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

    def calibrate(self, image_sets, cam, rows=8, columns=10, scaling=0.005, factor=2):
        """
        Calibrates a single camera of this stereoCamera instance.
        rows and columns need to be the real number of rows and columns in the chessboard-pattern

        image_sets are lists of lists of images, use one list of image for each video
        """
        # Only chessboard corners with all four sides being squares can be detected. (B W) Therefore the detectable
        # chessboard rows and columns are one less.                                  (W B)
        rows -= 1
        columns -= 1
        # Chessboard pixel coordinates (2D)
        imgpoints = []
        # Chessboard pixel coordinates in world-space (3D). Coordinate system defined by chessboard.
        objpoints = []

        for images in image_sets:
            # Generating the cutouts based on the camera-index
            images = [self(img)[cam] for img in images]
            # height and width are used in the calibration call
            height = images[0].shape[0]
            width = images[0].shape[1]
            # prepare object points, lower left corner of chessboard will be world coordinate (0, 0, 0)
            objp = np.zeros((columns * rows, 3), np.float32)
            objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
            objp = scaling * objp
            images = [cv2.resize(img, np.array(img.shape[:2])[::-1] * factor) for img in images]
            # Copies the scaled images for later usage
            images_old = np.array(images)
            # Drawing the cutout_corners
            cc = draw_cutout_corners(images[0], cam)
            offset = cc.min(axis=0)
            # Uses cc to generate image cutouts
            images = [img[cc[:, 1].min():cc[:, 1].max(), cc[:, 0].min():cc[:, 0].max()] for img in images]
            # Converts images to grayscale for chessboard_corner detection
            images = [cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) for img in images]
            # Shows one example from the image cutouts for evaluation
            img_show = np.array(images[0])
            cv2.putText(img_show,
                        f'Image section {cam} press any key to continue.',
                        (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (255, 255, 255),
                        1)
            cv2.imshow("Cutout", img_show)
            cv2.waitKey(0)
            cv2.destroyWindow("Cutout")
            # Tries to detect the corners
            corners = [cv2.findChessboardCorners(img, (rows, columns), None) for img in images]
            # Removes the images of unsuccessful corner-predictions. res[1] = corner-coordinates, res[0] = ret
            images = [images[i] for i, res in enumerate(corners) if res[0]]
            images_old = [images_old[i] for i, res in enumerate(corners) if res[0]]
            # Removes unsuccessful detections. res[1] = corner-coordinates, res[0] = ret
            corners = [res[1] for res in corners if res[0]]
            # skips following steps if now corners were detected
            if len(corners) == 0:
                continue

            # Parameter for chessboard detection refinement
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # Refines the corner predictions
            corners = [cv2.cornerSubPix(img, crnrs, (11, 11), (-1, -1), criteria)
                       for img, crnrs in zip(images, corners)]
            # Adjusts the corner-coordinates for the scaling and the cutout-offset.
            corners = [np.array([(np.array(coord) + offset) / factor for (coord) in crnrs], dtype=np.float32)
                       for crnrs in corners]
            # Shows one example of the drawn chessboard corners
            cv2.drawChessboardCorners(images_old[0],
                                      (rows, columns),
                                      corners[0] * factor,
                                      True)
            for i, [corner] in enumerate(corners[0]):
                cv2.putText(images_old[0],
                            f'{i}',
                            (int(corner[0] * factor),
                             int(corner[1]) * factor),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1,
                            (255, 255, 255),
                            1)

            cv2.imshow(f'Chessboard corners; Camera: {self}', images_old[0])
            key = cv2.waitKey(0)
            cv2.destroyWindow(f'Chessboard corners; Camera: {self}')
            objpoints.extend([objp for _ in corners])
            imgpoints.extend(corners)

        if imgpoints == []:
            print("No Corners were detected, failed calibration")
            return
        # Performs the calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(np.array(objpoints),
                                                           np.array(imgpoints),
                                                           (width, height),
                                                           None,
                                                           None)
        # Calculates the optimized camera matrix
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
        return

    def stereo_calibrate(self,
                         images,
                         rows=8,
                         columns=10,
                         scaling=0.005,
                         undistort=True,
                         stereocalibration_flags = cv2.CALIB_FIX_PRINCIPAL_POINT,
                         corner_detection=label_corners):
        """
        Performs stereo calibration for the stereoCamera instance
        """
        assert self.conf["camera_matrix"][0] is not None and self.conf["camera_matrix"][1] is not None, \
            "Calibrate both cameras first!"
        # open cv can only detect inner corners, so reducing the rows and columns by one
        rows -= 1
        columns -= 1
        # coordinates of squares in the checkerboard world space
        objp = np.zeros((rows * columns, 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
        objp = objp * scaling
        # Pixel coordinates in image space of checkerboard
        imgpoints_1 = []
        imgpoints_2 = []
        # World coordinates of the checkerboard. (World coordinate system has its origin in the bottom left corner of
        # the checkerboard)
        objpoints = []
        # Creates the two camera-outputs for each image in images
        images = [self(image) for image in images]
        if undistort:
            # Undistorts the images
            images = [[self.undistort_image(img1, 0), self.undistort_image(img1, 1)]
                      for img1, img2 in images]
        assert images[0][0].shape == images[0][1].shape, \
            "For both cameras must have the same resolution for stereo-calibration"
        # heigth and width will be used in stereocalibration
        height, width = images[0][1].shape[:2]
        # Rescales the images for better corner labeling
        factor = 3
        images = [[cv2.resize(img1, np.array(img1.shape[:2])[::-1] * factor),
                   cv2.resize(img2, np.array(img2.shape[:2])[::-1] * factor)]
                  for img1, img2 in images]
        # Copies the scales images for later usage
        images_old = np.array(images)
        # Chessboard corner detection or labeling
        corners = [[corner_detection(img1), corner_detection(img2)] for img1, img2 in images]
        # Letting the users switch the corners of img1 to match them with img2
        corners = [draw_cutout_corners(img1, img2, corners1, corners2) for [img1, img2], [corners1, corners2]
                   in zip(images, corners)]
        # rescaling the corners
        corners = [[corners1/factor, corners2/factor] for corners1, corners2 in corners]
        for corners1, corners2 in corners:
            objpoints.append(objp)
            imgpoints_1.append(corners1)
            imgpoints_2.append(corners2)
        # prerform stereo calibration on accumulated objectpoints
        imgpoints_1 = np.array(imgpoints_1, dtype=np.float32)
        imgpoints_1 = np.swapaxes(imgpoints_1, axis1=1, axis2=2)
        imgpoints_2 = np.array(imgpoints_2, dtype=np.float32)
        imgpoints_2 = np.swapaxes(imgpoints_2, axis1=1, axis2=2)
        objpoints = np.array(objpoints)
        objpoints = np.expand_dims(objpoints, axis=1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.00001)
        ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                                      imgpoints_1,
                                                                      imgpoints_2,
                                                                      np.array(self.conf["camera_matrix"][0]),
                                                                      np.array(self.conf["distortion"][0]),
                                                                      np.array(self.conf["camera_matrix"][1]),
                                                                      np.array(self.conf["distortion"][1]),
                                                                      (width, height),
                                                                      criteria=criteria,
                                                                      flags=stereocalibration_flags)

        
        optimized_camera_matrix1, roi = cv2.getOptimalNewCameraMatrix(CM1, np.array(self.conf["distortion"][0]), (width, height), 1,
                                                                   (width, height))
        optimized_camera_matrix2, roi = cv2.getOptimalNewCameraMatrix(CM2, np.array(self.conf["distortion"][1]), (width, height), 1,
                                                                   (width, height))
        # Matrix that rotates the coordinate system of the second camera to match the first.
        self.conf["rotation_matrix"][0] = R
        # Matrix that translates the coordinate system of the second camera to match the first.
        self.conf["translation_matrix"][0] = T
        self.conf["stereo_calibration_error"][0] = ret
        self.conf["camera_matrix"][0] = CM1
        self.conf["camera_matrix"][1] = CM2
        self.conf["optimized_camera_matrix"][0] = optimized_camera_matrix1
        self.conf["optimized_camera_matrix"][1] = optimized_camera_matrix2
        print(f'Stereo-calibration error: {ret}')
        print(f'Translation Matrix: {T}')
        print(f'Rotation Matrix: {R}')
        print(f"New Camera Matrix Mirror: {CM1}")
        print(f"New Camera Matrix Front: {CM2}")
        cv2.destroyAllWindows()
        return objpoints, imgpoints_1, imgpoints_2, images

    def set_anchor_point(self, img, cam):
        """
        Set an anchor point on the image using the mouse click.
        """
        anchor_point = None

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                nonlocal anchor_point
                anchor_point = (x, y)
                print("Left mouse button pressed!", anchor_point)
        while True:
            win_name = f"Set Anchor Point - Camera {cam}"
            instructions_text = f"Left-click where the {'mirror' if cam == 0 else 'frontal'}-camera should point. Press space when finished!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (10, 30)  # Bottom-left corner of the text in the image
            font_scale = 0.7
            color = (255, 255, 255)  # White color text
            thickness = 2
            img_with_text = img.copy()
            cv2.circle(img_with_text, anchor_point, 2, (0, 255, 0), 2)
            cv2.putText(img_with_text, instructions_text, org, font, font_scale, color, thickness)

            cv2.imshow(win_name, img_with_text)
            cv2.setMouseCallback(win_name, mouse_callback)
            key = cv2.waitKey(1)  # Wait indefinitely until a key is pressed
            if key & 0xFF == 32:
                cv2.destroyWindow(win_name)
                break
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

        return frame0, frame1

if __name__=="__main__":
    pass
