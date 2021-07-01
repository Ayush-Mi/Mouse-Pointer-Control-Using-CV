'''
setup openvino source environement by following command:
source /opt/intel/openvinso_2020.3.193/bin/setupvars.sh
'''
import cv2
import os
import sys
sys.path.append("/opt/intel/openvino_2020.3.194/python/python3.7/")
import numpy as np
import time
import math
from openvino.inference_engine import IENetwork,IECore
from mouse_controller import MouseController
from input_feeder import InputFeeder
from argparse import ArgumentParser
import logging
logging.basicConfig(filename='app.log', filemode='w')

from face_det_model import FaceDetectionClass
from face_landmark_model import FacialLandmarksClass
from gaze_model import GazeEstimationClass
from pose_model import HeadPoseEstimationClass

PATH_ = "/Users/amishra162/Documents/Nanodegree/submission/"
FACE_DET_MODEL = PATH_ + "intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001"
FACIAL_LANDMARK_MODEL = PATH_ + "intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009"
HEAD_POSE_MODEL = PATH_ + "intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001"
GAZE_MODEL = PATH_ + "intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002"
INPUT_VIDEO = PATH_ + 'starter/bin/demo.mp4'

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-fd", "--face_detection_model", required=False, default= FACE_DET_MODEL,
                        type=str,help="Face detection model path")
    parser.add_argument("-fl", "--facial_landmarks_model", required=False,default=FACIAL_LANDMARK_MODEL,
                        type=str,help="Facial Landmark detection model path")
    parser.add_argument("-hp", "--head_pose_model", required=False,default=HEAD_POSE_MODEL,
                        type=str,help="head pose estimation model path")
    parser.add_argument("-ge", "--gaze_estimation_model", required=False,default=GAZE_MODEL,
                        type=str,help="Gaze estimation model path")
    parser.add_argument("-i", "--input", required=False,default="cam", type=str,
                        help="Path to image or video file or CAM")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for face detection"
                             "(0.5 by default)")
    parser.add_argument("-vis", "--visualization_flag", required=False, default=True,
                        help="Visualization of output: True if required or else False")
    parser.add_argument("-speed", "--spd",type=str,required=False,default="medium",help="speed of the mouse pointer: fast, medium,slow")
    parser.add_argument("-precission", "--prc",type=str,required=False,default="high",help="speed of the mouse pointer: high,medium,low")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on:CPU, GPU, Myriad ")
    parser.add_argument("-control","--ctrl",type=bool,required=False,default=False,
                        help="True to enable mouse contorl with gaze or else False")
    return parser

args = build_argparser().parse_args()
input_file_path = args.input
input_feeder = InputFeeder(args.input)
input_feeder.load_data()

model_paths = {'Face_detection_model': args.face_detection_model,
                'Facial_landmarks_detection_model': args.facial_landmarks_model,
                'head_pose_estimation_model': args.head_pose_model,
                'gaze_estimation_model': args.gaze_estimation_model}

tok = time.time()
face_detection_model_object = FaceDetectionClass(model_name=args.face_detection_model,
    device=args.device, threshold=args.prob_threshold,
    extensions=None)
face_detection_model_object.load_model()
tok_end = time.time()

tik = time.time()
facial_landmarks_detection_model_object = FacialLandmarksClass(
    model_name=args.facial_landmarks_model,
    device=args.device, extensions=None)
facial_landmarks_detection_model_object.load_model()
tik_end = time.time()

tik_2 = time.time()
gaze_estimation_model_object = GazeEstimationClass(
    model_name=args.gaze_estimation_model, device=args.device, extensions=None)
gaze_estimation_model_object.load_model()
tik_2_end = time.time()

tik_3 = time.time()
head_pose_estimation_model_object = HeadPoseEstimationClass(
    model_name=args.head_pose_model, device=args.device, extensions=None)
head_pose_estimation_model_object.load_model()
tik_3_end = time.time()

logging.info("Time to load Face Detection Model: {}".format(tok_end-tok))
logging.info("Time to load Facial Landmark Detection Model: {}".format(tik_end-tik))
logging.info("Time to load Gaze Estimation Model: {}".format(tik_2_end-tik))
logging.info("Time to load Head Pose Estimation Model: {}".format(tik_3_end-tik))
logging.info("Total time taken for loading model: {}".format(time.time()-tok))


mouse_controller_object = MouseController(args.prc,args.spd)

while True:
    for flag, frame in input_feeder.next_batch():
        start_time = time.time()
        if not flag:
            break
        face_coordinates, face_image = face_detection_model_object.predict(frame.copy())

        if face_coordinates == 0:
            continue
        head_pose_estimation_model_output = head_pose_estimation_model_object.predict(face_image)

        left_eye_image, right_eye_image, eye_coord = facial_landmarks_detection_model_object.predict(face_image)
        #print(face_image.shape)
        #print(eye_coord)

        mouse_coordinate, gaze_vector = gaze_estimation_model_object.predict(left_eye_image, right_eye_image,
                                                                             head_pose_estimation_model_output)
        if args.ctrl:
            mouse_controller_object.move(mouse_coordinate[0], mouse_coordinate[1])

        print("Time to process one frame: {}".format(time.time()-start_time))
        logging.info("Time to process one frame: {}".format(time.time()-start_time))

        if args.visualization_flag:

            tmp_image = frame.copy()

            cv2.rectangle(tmp_image, (face_coordinates[0], face_coordinates[1]),
                                    (face_coordinates[2], face_coordinates[3]), (0, 150, 0), 3)

            cv2.rectangle(tmp_image, ((eye_coord[0][0]+face_coordinates[0]), 
                                    (eye_coord[0][1]+face_coordinates[1])), 
                                    ((eye_coord[0][2]+face_coordinates[0]),
                                    (eye_coord[0][3]+face_coordinates[1])),
                                (150, 0, 150))
            cv2.rectangle(tmp_image, ((eye_coord[1][0]+face_coordinates[0]), 
                                    (eye_coord[1][1]+face_coordinates[1])), 
                                    ((eye_coord[1][2]+face_coordinates[0]), 
                                    (eye_coord[1][3]+face_coordinates[1])),
                                (150, 0, 150))

            cv2.putText(tmp_image,
                                "yaw:{:.1f} | pitch:{:.1f} | roll:{:.1f}".format(head_pose_estimation_model_output[0],
                                                                                head_pose_estimation_model_output[1],
                                                                                head_pose_estimation_model_output[2]),
                                (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.35, (0, 0, 0), 1)
            
            img_hor = cv2.resize(tmp_image, (frame.shape[1], frame.shape[0]))

            cv2.imshow('Visualization', tmp_image)
            if cv2.waitKey(27):
                break


input_feeder.close()
cv2.destroyAllWindows()