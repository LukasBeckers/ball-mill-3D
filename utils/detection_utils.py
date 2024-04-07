import cv2
import os
from ultralytics import YOLO

from camera_utils import *





class ballDetector():
    """
    Refines the detection results of the trained YOLOV8
    detector
    """
    def __init__(self):
        self.ROI = None
        self.region_size = 40
        self.warmup_regions = []
        self.warmup_steps = 5
        self.current_warmup_steps = 0

    def _warmup(self, detection_results):
        """
        This function can be used to automatically determine the starting 
        regions of intrest for the ball/balls.
        It works by removing outliers from the first self.warmup_steps 
        detections and using the last valid detection as starting ROI.
        """
        detection_results = detection_results[0]  # We only use single frame detection.
        boxes = detection_results.boxes.xyxy.tolist()
        classes = detection_results.boxes.cls

        for cl, box in zip(classes, boxes):
            if  cl == 0:  # 0 = Ball, 1 = Red Sticker
                coord = [box[0] + box[2], box[1] + box[3] ] #Box = xyxy creating average coord 
                self.warmup_regions.append(coord)

    def __call__(self, detection_results):
        """
        detection_results = YOLOV8 results from detector trained
        to detect the ball and the red stickers. 
        """
        
        boxes = detection_results.boxes.xyxy
        classes = detection_results.boxes.cls

        print("Boxes", boxes, "Classes", classes)



if __name__=="__main__":

    # Load a model
    #model = YOLO("yolov8n.yaml")  # build a new model from scratch
    #model = YOLO("./runs/detect/train2/weights/best.pt")  # load a pretrained model (recommended for training)

    # Use the model
    #model.train(data="config.yaml", epochs=150)  # train the model
    #raise EOFError
    ballD = ballDetector()
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
    vL = videoLoader()
    Dt1 = ballDetector()   
    Dt2 = ballDetector()
    files = os.listdir("../videos")
    model = YOLO("../weights/best.pt")

    for i, file in enumerate(files[6:]):
        vL.load_video(f"../videos/{file}")
        frames = vL[100:-100]

        for j, frame in enumerate(frames):
            frame1, frame2 = sC(frame)
            results1 = model(frame1)
            results2 = model(frame2)
            Dt1._warmup(results1)
            Dt2._warmup(results2)
            # for det in results2:
            #    print("Boxes" * 10)
            #    print("Boxes", det.boxes.xyxy, det.boxes.cls)
            #    cv2.imshow("Frame2", frame2)
            #    cv2.waitKey(1)

                
            # Draw the detections on frame1 and frame2.
            # This assumes the results object has a method to render the detections which returns the image.
            #frame1_with_detections = results1.render()[0]
            #frame2_with_detections = results2.render()[0]

            # Write the frame into the file 'output_cam1.mp4' and 'output_cam2.mp4'
            #out_cam1.write(frame1)
            #out_cam2.write(frame2)
            #out_cam.write(frame)

        #out_cam1.release()
        #out_cam2.release()
        #out_cam.release()




