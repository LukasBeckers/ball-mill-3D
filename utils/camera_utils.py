from utils.video_utils import *
import json


class stereoCamera():
    def __init__(self, camera_size=(480, 620), anchor_cam1=None, anchor_cam2=None):
        self.cam_size = camera_size
        self.anchor_points = {0:np.array(anchor_cam1), 1:np.array(anchor_cam2)}
        self.fisheye = False

    def undistort_image(self, img):
        """
        Undistorts an image using the camera_matrix and the distortion values obtained by camera calibration.
        :param img:             Image to undistort
        :param camera_matrix:   Camera matrix obtained by camera claibration
        :param distortion:      Distortion parameters obtained by camera calibration.
        :param optimized
        _camera_matrix:         Camera matrix optimized by cv2.getOptoimalNewCameraMatrix

        :return:                Undistorted image, and new camera_matrix
        """
        if self.fisheye:
            img = self.undistort_fisheye(img)
            return img

        img = cv2.undistort(img, self.camera_matrix, self.distortion, None, self.optimized_camera_matrix)
        return img

    def undistort_fisheye(self, img):
        h, w = img.shape[:2]
        dim = img.shape[:2][::-1]
        # maybe I will put the distortion maps in the config files, but for now this quick and dirty fix will do.
        # the maps are stored, because the initUndistortRectifyMap metod call takes about 0.22 seconds.
        try:
            map1 = self.fisheye_map1
            map2 = self.fisheye_map2
        except Exception:
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.fisheye_camera_matrix,
                                                             self.fisheye_distortion, np.eye(3),
                                                             self.fisheye_optimized_camera_matrix,
                                                             [w, h],
                                                             cv2.CV_16SC2)
            self.fisheye_map1 = map1
            self.fisheye_map2 = map2
        img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return img

    def set_config(self, config_name, value):
        try:
            with open('camera_configuration.json', 'r') as json_file:
                configurations = json.load(json_file)
        except FileNotFoundError:
            print("No camera configuration file found!")
            return

        # converting np arrays to lists
        if isinstance(value, (np.ndarray, list, tuple)):
            value = np.array(value).tolist()

        # storing the config value of the camera in a JSON format.
        configurations[self.name][config_name] = value

        # saving the updated dict as JSON
        with open('camera_configuration.json', 'w') as json_file:
            json.dump(configurations, json_file)

    def get_config(self, config_name, value):
        try:
            with open('camera_configuration.json', 'r') as json_file:
                configurations = json.load(json_file)
        except FileNotFoundError:
            print("No camera configurations file found!")
            return

        # storing the stream of the camera in a JSON format.
        value = configurations[self.name][config_name]

        return value

    def load_configs(self):
        """
        If a camera with the same name was already configured, there will probably be a JSON file with
        configurations and calibrations for this setup, it will be loaded by this function.
        """
        try:
            with open('camera_configuration.json', 'r') as json_file:
                configurations = json.load(json_file)
                self.stream = configurations[self.name]["stream"]
                try:
                    self.resolution  = configurations[self.name]["resolution"]
                except KeyError: # Going with standard resolution 480 * 640.
                    self.resolution = False

                # trying to load attributes that are only added to the camera during calibration.
                try:
                    self.projection_error = configurations[self.name]["projection_error"]
                    self.camera_matrix = np.array(configurations[self.name]["camera_matrix"])
                    self.distortion = np.array(configurations[self.name]["distortion"])
                    self.optimized_camera_matrix = np.array(configurations[self.name]["optimized_camera_matrix"])

                    self.current_camera_matrix = self.optimized_camera_matrix  # gets overwritten if fisheye
                    self.calibrated = True
                except KeyError as E:
                    print('Could not load all values', E)
                    self.calibrated = False
                # trying to load fisheye calibration
                # fisheye calibration is optional and will only throw an error if fisheye is set to true in config.
                # and not all params could be loaded
                try:
                    self.fisheye = configurations[self.name]["fisheye"]

                    if self.fisheye:
                        self.fisheye_camera_matrix = np.array(configurations[self.name]["fisheye_camera_matrix"])
                        self.fisheye_distortion = np.array(configurations[self.name]["fisheye_distortion"])
                        self.fisheye_optimized_camera_matrix = np.array(
                            configurations[self.name]["fisheye_optimized_camera_matrix"])

                        self.current_camera_matrix = self.fisheye_optimized_camera_matrix
                                                                                # if the camera is fisheye calibrated
                                                                                # the current camera matrix is
                                                                                # overwritten by the
                        #                                                         fisheye_camera_matrix
                except KeyError as E:
                    if self.fisheye:
                     print("Error while loading fisheye calibration, maybe camera was not fisheye calibrated jet", E)

            return True

        except FileNotFoundError:
            print("No camera_configuration.json file found!")
            return False


    def calibrate(self, images, cam, rows=8, columns=10, scaling=0.005):
        """
        Thanks to: https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html

        # Calibration for normal cameras (pin hole)

        Calculates the camera matrix for a camera based on a checkerboard-pattern.
        :param rows:        Number of rows in chessboard-pattern.
        :param columns:     Number of columns in chessboard-pattern.
        :param scaling:     Realworld size of a chess-board square to scale the coordinate system.
                            I will try to keep all units in meters so a good initial value for this will be 0.01 or 1 cm

        :n_images:          Number of photos that will be taken for calibration.
        :param fisheye:     Indicates if fisheye calibration should be used before normal calibration. Config value is
                            used by default
        :return:
        """

        images = [self(img)[cam] for img in images]
        print("Len Images", len(images))

        # Only chessboard corners with all four sides being squares can be detected. (B W) Therefore the detectable
        # chessboard is one smaller in number of rows and columns.                   (W B)
        rows -= 1
        columns -= 1
        # termination criteria
        # If no chessboard-pattern is detected, change this... Don't ask me what to change about it!
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, lower left corner of chessboard will be world coordinate (0, 0, 0)
        objp = np.zeros((columns * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
        objp = scaling * objp

        # Chessboard pixel coordinates (2D)
        imgpoints = []
        # Chessboard pixel coordinates in world-space (3D). Coordinate system defined by chessboard.
        objpoints = []

        for i, img in enumerate(images):
            print(i)
            img_old = np.array(img)

            cv2.imshow(f'Camera: {self}', img)
            k = cv2.waitKey(0)

            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            # localizing the chessboard corners in image-space.
            ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)
            print("ret", ret)
            if ret:
                # trying to improve the detection of the chessboard corners!
                corners = cv2.cornerSubPix(gray,
                                                       corners,
                                                       (11, 11),  # size of kernel!
                                                       (-1, -1),
                                                       criteria)
                cv2.drawChessboardCorners(img, (rows, columns), corners, ret)
                for i, [corner] in enumerate(corners): # Check if enumeration is consistent
                    cv2.putText(img, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX, 1,
                                            (0, 0, 255), 1)
                cv2.imshow(f'Chessboard corners; Camera: {self}', img)
                key = cv2.waitKey(0)
                if key & 0xFF == ord('s'):  # press s to switch the ordering of the corners
                    cv2.destroyWindow(f'Chessboard corners; Camera: {self}')
                    corners = corners[::-1]
                    # drawing the new corners
                    cv2.drawChessboardCorners(img_old, (rows, columns), corners, ret)
                    for i, [corner] in enumerate(corners):  # Check if enumeration is consistent
                        cv2.putText(img_old, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX, 1,
                                                (0, 0, 255), 1)
                    cv2.imshow(f'Chessboard corners; Camera: {self}', img_old)
                    cv2.waitKey(0)
                    cv2.destroyWindow(f'Chessboard corners; Camera: {self}')
                    objpoints.append(objp)
                    imgpoints.append(corners)


        height = img.shape[0]
        width = img.shape[1]

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
        # saving the optimized camera matrix
        height, width = img.shape[:2]
        optimized_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1,
                                                                   (width, height))

        print('rmse:', ret)
        print('camera matrix:\n', mtx)
        print('optimized camera matrix:\n', optimized_camera_matrix)
        print('distortion coeffs:\n', dist)

        self.projection_error = ret
        self.camera_matrix = mtx
        self.optimized_camera_matrix = optimized_camera_matrix
        self.distortion = dist

        # Updating the camera_configuration.json file
        self.set_config(f"projection_error_{cam}", self.projection_error)
        self.set_config(f"camera_matrix_{cam}", self.camera_matrix)
        self.set_config(f"distortion_{cam}", self.distortion)
        self.set_config(f"optimized_camera_matrix_{cam}", self.optimized_camera_matrix)

        # closing the window after calibration
        cv2.destroyWindow(f'Camera: {self}')
        self.calibrated = True
        cv2.destroyAllWindows()
        return


    def set_anchor_point(self, img, cam):
        """
        img: frame from Video
        cam: number of camera for which to set the anchor-point (0 or 1)
        """
        global anchor_point
        def mouse_callback(event, x, y, flags, param):
            global anchor_point
            if event == cv2.EVENT_LBUTTONDOWN:
                anchor_point = (x, y)
                print("Left mouse button pressed!", anchor_point)
        win_name = "Set Anchor Point" + str(cam)
        cv2.imshow(win_name, img)
        cv2.setMouseCallback(win_name, mouse_callback)
        cv2.waitKey(0)
        self.anchor_points[cam] = anchor_point
        cv2.destroyWindow(win_name)


    def draw_camera_region(self, img):

        # Drawing camera 0
        anchor_point = np.array(self.anchor_points[0])
        start_point = anchor_point - np.array(self.cam_size)/2
        end_point = anchor_point + np.array(self.cam_size)/2
        img = cv2.rectangle(img,  start_point.astype(np.int32), end_point.astype(np.int32), (255, 0, 0), 5)

        # Drawing camera 1
        anchor_point = np.array(self.anchor_points[1])
        start_point = anchor_point - np.array(self.cam_size) / 2
        end_point = anchor_point + np.array(self.cam_size) / 2
        img = cv2.rectangle(img, start_point.astype(np.int32), end_point.astype(np.int32), (255, 0, 0), 5)

        return img

    def __call__(self, image):

        # Camera0
        anchor_point0 = np.array(self.anchor_points[0])
        start_point0 = anchor_point0 - np.array(self.cam_size) / 2
        end_point0 = anchor_point0 + np.array(self.cam_size) / 2

        # Camera 1
        anchor_point1 = np.array(self.anchor_points[1])
        start_point1 = anchor_point1 - np.array(self.cam_size) / 2
        end_point1 = anchor_point1 + np.array(self.cam_size) / 2

        # checking for negative values and adjusting the anchor size
        for i, val in enumerate(start_point0):
            if val < 0:
                self.anchor_points[0][i] -= val
                return self(image)

        for i, val in enumerate(start_point1):
            if val < 0:
                self.anchor_points[1][i] -= val
                return self(image)

        frame0 = image[int(start_point0[1]): int(end_point0[1]), int(start_point0[0]): int(end_point0[0])]
        frame1 = image[int(start_point1[1]): int(end_point1[1]), int(start_point1[0]): int(end_point1[0])]

        return frame0, frame1

if __name__=="__main__":
    sC = stereoCamera(camera_size=(400, 250), anchor_cam1=(764, 136), anchor_cam2=(735, 324))

    vL = videoLoader()
    vL.load_video("../videos/VID-20240329-WA0015.mp4", start_frame=100, end_frame=-100)
    frame = vL[10]
    #sC.set_anchor_point(frame, 0)
    #sC.set_anchor_point(frame, 1)

    frames = vL[:10]

    for frame in frames[:1]:
        print(sC.anchor_points)
        #frame = sC.draw_camera_region(frame)
        #cv2.imshow("Frame", frame)
        frame0, frame1 = sC(frame)
        cv2.imshow("frame0", frame0)
        cv2.imshow("frame1", frame1)
        cv2.waitKey(0)

    sC.calibrate(frames, 0)
    sC.calibrate(frames, 1)