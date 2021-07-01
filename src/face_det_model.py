import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork,IECore
import math
import logging
logging.basicConfig(filename='app.log', filemode='w')
from model import load_model

#Face Detection Object
class FaceDetectionClass:
    '''
    Reference:
    Below code has been taught in Module 2 - Lesson 4
    The Inference Engine
    '''
    def __init__(self, model_name, device, threshold, extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_bin = model_name + '.bin'
        self.model_xml = model_name + ".xml"
        self.device = device
        self.threshold = threshold
        self.extension = extensions
        self.face_cropped = None
        self.first_detected_face = None
        self.face_co = None
        self.results = None
        self.initial_img = None
        self.net = None
        self.model = IENetwork(self.model_xml, self.model_bin)
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.net = load_model(self.model_xml,self.model_bin,self.device)


    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        self.initial_img = self.preprocess_input(image)
        self.results = self.net.infer({self.input_name: self.initial_img})
        self.face_co = self.preprocess_output(self.results, image)

        if len(self.face_co) == 0:
            print("No face detected")
            logging.error("No Face is detected, Next frame will be processed..")

            return 0, 0

        self.first_detected_face = self.face_co[0]
        face_cropped = image[self.first_detected_face[1]:self.first_detected_face[3],
                             self.first_detected_face[0]:self.first_detected_face[2]]
        return self.first_detected_face, face_cropped

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        pre_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        pre_frame = pre_frame.transpose((2, 0, 1))
        pre_frame = pre_frame.reshape(1, *pre_frame.shape)

        return pre_frame

    def preprocess_output(self, outputs,image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        face_co = []
        outs = outputs[self.output_name][0][0]
        for box in outs:
            conf = box[2]
            if conf >= self.threshold:
                xmin = int(box[3] * image.shape[1])
                ymin = int(box[4] * image.shape[0])
                xmax = int(box[5] * image.shape[1])
                ymax = int(box[6] * image.shape[0])
                face_co.append([xmin, ymin, xmax, ymax])
        return face_co
