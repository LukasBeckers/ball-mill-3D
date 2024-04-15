from utils.video_utils import *
from collections import defaultdict


class stereoCamera():
    def __init__(self, name="", **kwargs):
        """
        Possible **kwargs are all dicts with the keys = int(camera_index) ie. 0 or 1:

        camera_size
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

    def __str__(self):
        return self.name


    def undistort_image(self, img, cam):
        """
        Undistorts an image using the camera_matrix and the distortion values obtained by camera calibration.

        :return:                Undistorted image
        """
        img = cv2.undistort(img, self.conf["camera_matrix"][cam], self.conf["distortion"][cam], None,
                            self.conf["optimized_camera_matrix"][cam])
        return img

    def calibrate(self, images, cam, rows=8, columns=10, scaling=0.005):

        images = [self(img)[cam] for img in images]

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
            img_old = np.array(img)
            factor = 4

            img = cv2.resize(img, np.array(img.shape[:2])[::-1] * factor)

            global cutout_corners # will be deleted after iteration over images
            try: # fuck it!
                _ = len(cutout_corners)
            except Exception:
                cutout_corners = []

            def mouse_callback(event, x, y, flags, param):
                global cutout_corners
                if event == cv2.EVENT_LBUTTONDOWN:
                    cutout_corners.append((x, y))
                    print("Left mouse button pressed!", cutout_corners)
            if len(cutout_corners) <2:
                while True:
                    for point in cutout_corners[-2:]:
                        cv2.circle(img, point, 2, (0,255, 0), 2)
                    cv2.putText(img, f'Click on the corners of the chess board to cut out the image section', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1,(255, 255, 255), 1)
                    cv2.imshow(f'Camera: {cam}', img)
                    cv2.setMouseCallback(f"Camera: {cam}", mouse_callback)
                    key = cv2.waitKey(1)
                    if key & 0xFF == 32:
                        cv2.destroyWindow(f"Camera: {cam}")
                        break
            self.cutout_corners = cutout_corners[-2:]

            cutout_corners = self.cutout_corners
            cc_arr = np.array(cutout_corners[-2:], dtype=np.int32)

            img = img[cc_arr[:, 1].min() : cc_arr[:, 1].max(), cc_arr[:,0].min() : cc_arr[:,0].max()]
            offset = cc_arr.min(axis=0)

            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

            cv2.putText(img, f'Image section {i} press any key to continue.', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1,(255, 255, 255), 1)
            cv2.imshow("Cutout", img)
            cv2.waitKey(0)
            cv2.destroyWindow("Cutout")

            gray = img
            # localizing the chessboard corners in image-space.
            ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)
            print("ret", ret)
            if ret:
                # trying to improve the detection of the chessboard corners!
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                # adjusting the corner coordinates for the scaling and the cutout
                corners = np.array([(np.array(coord) + offset) / factor for (coord) in corners], dtype=np.float32)

                img = np.array(img_old)
                # resizing again to properly display the corners
                img = cv2.resize(img, np.array(img.shape[:2])[::-1] * factor)
                cv2.drawChessboardCorners(img, (rows, columns), corners*factor, ret)
                for i, [corner] in enumerate(corners):
                    cv2.putText(img, f'{i}', (int(corner[0]*factor), int(corner[1])*factor), cv2.FONT_HERSHEY_COMPLEX,
                                1, (0, 0, 255), 1)

                cv2.imshow(f'Chessboard corners; Camera: {self}', img)
                key = cv2.waitKey(0)
                if key & 0xFF == ord('s'):  # press "s" to switch the ordering of the corners
                    cv2.destroyWindow(f'Chessboard corners; Camera: {self}')
                    corners = corners[::-1]

                    # drawing the new corners
                    img = np.array(img_old)
                    img = cv2.resize(img, np.array(img.shape[:2])[::-1] * factor)
                    cv2.drawChessboardCorners(img, (rows, columns), corners*factor, ret)
                    for i, [corner] in enumerate(corners):
                        cv2.putText(img, f'{i}', (int(corner[0] * factor), int(corner[1]) * factor),
                                    cv2.FONT_HERSHEY_COMPLEX, 1,
                                    (0, 0, 255), 1)
                    cv2.imshow(f'Chessboard corners; Camera: {self}', img)
                    cv2.waitKey(0)
                    cv2.destroyWindow(f'Chessboard corners; Camera: {self}')
                objpoints.append(objp)
                imgpoints.append(corners)
        del cutout_corners
        if imgpoints == []:
            print("No Corners were detected, failed calibration")
            return

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

        self.conf["projection_error"][cam] = ret
        self.conf["camera_matrix"][cam] = mtx
        self.conf["optimized_camera_matrix"][cam] = optimized_camera_matrix
        self.conf["distortion"][cam] = dist

        # closing the window after calibration
        cv2.destroyAllWindows()
        return

    def stero_calibrate(self, images, rows=8, columns=10, scaling=0.005):
        """

        """
        assert self.conf["camera_matrix"][0] is not None and self.conf["camera_matrix"][1] is not None, \
            "Calibrate both cameras first!"

        def draw_lines(img):
            global line # line = (x1, y1, x2, y2)
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
                    print("Escaping Line drawing")
                    break
                if len(line) == 4:
                    lines.append(line)
                    line = []
                    img = np.array(img_old)
                    for l in lines:
                        cv2.line(img, (l[0], l[1]), (l[2], l[3]), color=(0, 255, 0), thickness=1)
                    print("Line drawn")

            return img, line

        def line_intersection(lines):
            """Finds the intersection of two lines given in Hesse normal form."""

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

        # open cv can only detect inner corners, so reducing the rows and columns by one
        rows -= 1
        columns -= 1

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

        # coordinates of squares in the checkerboard world space
        objp = np.zeros((rows * columns, 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
        objp = scaling * objp

        # Pixel coordinates in image space of checkerboard
        imgpoints_1 = []
        imgpoints_2 = []

        # World coordinates of the checkerboard. (World coordinate system has its origin in the bottom left corner of
        # the checkerboard.
        objpoints = []

        for img in images:
            img1, img2 = self(img)

            assert img1.shape == img2.shape, "For both cameras must have the same resolution for stereo-calibration"

            height, width = img1.shape[:2]

            factor = 4

            img1 = cv2.resize(img1, np.array(img1.shape[:2])[::-1] * factor)
            img2 = cv2.resize(img2, np.array(img2.shape[:2])[::-1] * factor)
            img1_old = np.array(img1)
            img2_old = np.array(img2)
            img1, lines1 = draw_lines(img1)
            img2, lines2 = draw_lines(img2)

            corners1 = line_intersection(lines1)
            corners2 = line_intersection(lines2)


            # Ignoring intersections outside the image
            w, h = img1.shape[:2]

            corners1 = np.array([[corner] for corner in corners1 if h > corner[0] > 0 and w > corner[1] > 0])
            corners2 = np.array([[corner] for corner in corners2 if h > corner[0] > 0 and w > corner[1] > 0])

            for i, [corner] in enumerate(corners1):
                cv2.putText(img1, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255), 1)
            for i, [corner] in enumerate(corners2):
                cv2.putText(img2, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255), 1)
            cv2.imshow(f'Detection 1', img1)
            cv2.imshow(f'Detection 2', img2)
            key = cv2.waitKey(0)

            if key & 0xFF == ord('s'):  # press s to switch ordering of img1
                cv2.destroyWindow(f'Detection 1')
                print("Corners1 before", corners1.shape)
                corners1 = corners1[::-1]
                print("Corners1 after", corners1.shape)
                # drawing the new corners
                img1 = np.array(img1_old)
                for i, [corner] in enumerate(corners1):
                    cv2.putText(img1, f'{i}', (int(corner[0]), int(corner[1])),
                                cv2.FONT_HERSHEY_COMPLEX,
                                1,
                                (0, 0, 255), 1)
                cv2.imshow(f'Detection 1', img1)
                cv2.waitKey(0)

            # adjusting corner coordinates for scaling
            corners1 /= factor
            corners2 /= factor
            objpoints.append(objp)

            imgpoints_1.append(corners1)
            imgpoints_2.append(corners2)
            cv2.destroyWindow(f'Detection 1')
            cv2.destroyWindow(f'Detection 2')
        # prerform stereo calibration on accumulated objectpoints
        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC

        imgpoints_1 = np.array(imgpoints_1, dtype=np.float32)
        imgpoints_1 = np.swapaxes(imgpoints_1, axis1=1, axis2=2)
        imgpoints_2 = np.array(imgpoints_2, dtype=np.float32)
        imgpoints_2 = np.swapaxes(imgpoints_2, axis1=1, axis2=2)
        objpoints = np.array(objpoints)
        objpoints = np.expand_dims(objpoints, axis=1)
        print("objectpoints shape", objpoints.shape)
        print("imgpoints shape", imgpoints_1.shape)
        ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                                      imgpoints_1,
                                                                      imgpoints_2,
                                                                      self.conf["camera_matrix"][0],
                                                                      self.conf["distortion"][0],
                                                                      self.conf["camera_matrix"][1],
                                                                      self.conf["distortion"][1],
                                                                      (width, height),
                                                                      criteria=criteria,
                                                                      flags=stereocalibration_flags)

        # Matrix that rotates the coordinate system of the second camera to match the first.
        self.conf["rotation_matrix"][0] = R
        # Matrix that translates the coordinate system of the second camera to match the first.
        self.conf["translation_matrix"][0] = T
        self.conf["stereo_calibration_error"][0] = ret


        print(f'Stereo-calibration error: {ret}')
        print(f'Translation Matrix: {T}')
        print(f'Rotation Matrix: {R}')


        cv2.destroyAllWindows()
        return

    def set_anchor_point(self, img, cam):
        """
        Set an anchor point on the image using the mouse click.
        
        Parameters:
        img (numpy.ndarray): Frame from video.
        cam (int): Number of the camera (0 or 1) for which to set the anchor point.
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

        return frame0, frame1

if __name__=="__main__":

    sC = stereoCamera(camera_size={0: (480, 240), 1: (480, 240)},
                      anchor_point={0: (587, 269), 1: (598, 433)},
                      camera_matrix={0: np.array([[2.24579312e+03, 0.00000000e+00, 6.06766474e+02],
                                                  [0.00000000e+00, 3.18225724e+03, 2.87228912e+02],
                                                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
                                     1: np.array([[9.17450924e+02, 0.00000000e+00, 5.97492459e+02],
                                                  [0.00000000e+00, 1.08858369e+03, 2.96145751e+02],
                                                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])},
                      optimized_camera_matrix={0: np.array([[1.98885152e+03, 0.00000000e+00, 5.83904948e+02],
                                                            [0.00000000e+00, 2.71756632e+03, 3.41261625e+02],
                                                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
                                               1: np.array([[9.35319179e+02, 0.00000000e+00, 5.90025655e+02],
                                                            [0.00000000e+00, 1.09136910e+03, 2.97696817e+02],
                                                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])},
                      projection_error={0: 0.26768362133770185, 1: 0.29408707559840946},
                      distortion={
                          0: np.array(
                              [[-1.53486495e+00, 1.95803727e+01, 1.63594781e-01, -2.81574724e-02, -1.10093707e+02]]),
                          1: np.array([[0.03667417, 0.10305058, 0.00557331, -0.00655738, -0.16404791]])},
                      stereo_calibration_error={0: 0.6988643727550614},
                      translation_matrix={0: [[-0.60330682], [-0.39384531], [1.07405106]]},
                      rotation_matrix={0: [[0.73971458, 0.1145444, 0.66310023],
                                           [-0.09028238, - 0.95960383, 0.26647622],
                                           [0.66683688, - 0.25698261, - 0.69949161]]}
                      )
    sC = stereoCamera(camera_size={0: (480, 240), 1: (480, 240)})
    vL = videoLoader()
    vL.load_video("../videos/WhatsApp Video 2024-04-02 at 6.40.29 PM.mp4")
    frame = vL[10]
    sC.set_anchor_point(frame, 0)
    sC.set_anchor_point(frame, 1)
    frames = vL[:2]

    for frame in frames[:1]:
        frame = sC.draw_camera_region(frame)
        cv2.imshow("Frame", frame)
        frame0, frame1 = sC(frame)
        cv2.imshow("frame0", frame0)
        cv2.imshow("frame1", frame1)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    #sC.calibrate(frames, 0)
    #sC.calibrate(frames, 1)
    #sC.stero_calibrate(frames)
