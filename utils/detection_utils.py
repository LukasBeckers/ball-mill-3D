import cv2

from camera_utils import *

class blobDetector():
    pass
    def __init__(self):
        params = cv2.SimpleBlobDetector_Params()

        params.filterByColor = True
        params.blobColor = 255  # Assuming white blobs

        params.filterByArea = True
        params.minArea = 20  # Adjust based on the size of the ball in your image

        params.filterByCircularity = True
        params.minCircularity = 0.8  # Adjust for how round the ball is
        # Create a detector with the parameters
        self.detector = cv2.SimpleBlobDetector_create(params)

    def __call__(self, img):
        keypoints = self.detector.detect(img)
        return keypoints
class ballDetector():
    def __init__(self):
        self.ROI = None  #[x1, y1, x2, y2]
        self.region_size = 40


    def _set_roi(self, img):
        img = np.array(img)
        global roi_center
        roi_center=[]
        def mouse_callback(event, x, y, flags, param):
            global roi_center
            if event == cv2.EVENT_LBUTTONDOWN:
                roi_center = (x, y)
                print("Left mouse button pressed!", roi_center)

        cv2.putText(img,
                    "Click the ball, and confirm with Space",
                    (5, 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2, cv2.LINE_AA)
        
        while len(roi_center) == 0:
            cv2.imshow("Kugel", img)
            cv2.setMouseCallback("Kugel", mouse_callback)
            cv2.waitKey(1)

        roi = np.array([roi_center[0] - self.region_size, roi_center[1] - self.region_size,
                        roi_center[0] + self.region_size, roi_center[1] + self.region_size], dtype=np.int32)

        self.ROI = roi


    def __call__(self, img):
        if self.ROI is None:
            self._set_roi(img)
        else:
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            img = img[self.ROI[1]:self.ROI[3], self.ROI[0]:self.ROI[2]]

            circles = cv2.HoughCircles(img,
                                       cv2.HOUGH_GRADIENT,
                                       1,
                                       10,
                                       param1=100,
                                       param2=20,
                                       minRadius=5,
                                       maxRadius=25)
            if circles is None:
                self.ROI = None
                return self(img)

            circles = [[[x + self.ROI[0], y + self.ROI[1], z]] for [x, y, z] in circles[0]]
            
            roi_center = circles[0][0] # first circle will be used as the main detection
            self.ROI = np.array([roi_center[0] - self.region_size, roi_center[1] - self.region_size,
                        roi_center[0] + self.region_size, roi_center[1] + self.region_size], dtype=np.int32)
            
            return circles
class stickerDetector():
    pass





if __name__=="__main__":
    ballD = ballDetector()
    sC = stereoCamera(camera_size={0: (300, 150), 1: (300, 150)},
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
    vL = videoLoader()
    vL.load_video("../videos/WhatsApp Video 2024-03-29 at 19.13.45.mp4", start_frame=100, end_frame=-100)
    # frame = vL[10]
    # sC.set_anchor_point(frame, 0)
    # sC.set_anchor_point(frame, 1)
    frames = vL[100:]

    for frame in frames:
        frame = sC.draw_camera_region(frame)

        frame0, frame1 = sC(frame)
        # Convert the image from BGR to HSV color space

        kp = ballD(frame1)
        print("KP", kp)
        if kp is None: continue

        for circle in kp[0]:
            print(circle)
            cv2.circle(frame1, (int(circle[0]), int(circle[1])), 1, (0, 255 ,0), 3)
            cv2.circle(frame1, (int(circle[0]), int(circle[1])), int(circle[2]), (0, 255, 0), 3)
        cv2.imshow("frame1", frame1)

        cv2.waitKey(0)
    cv2.destroyAllWindows()
