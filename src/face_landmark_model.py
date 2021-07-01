import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork,IECore
import math
import logging
logging.basicConfig(filename='app.log', filemode='w')
#Face Landmark Detection Model
from model import load_model

class FacialLandmarksClass:
    '''
    Reference:
    Below code has been taught in Module 2 - Lesson 4
    The Inference Engine
    '''

    def __init__(self, model_name, device, extensions=None):
        '''
        this method is to set instance variables.
        '''
        self.model_bin = model_name + '.bin'
        self.model_xml = model_name + '.xml'
        self.device = device
        self.extension = extensions
        self.model = IENetwork(self.model_xml, self.model_bin)
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.net = load_model(self.model_xml,self.model_bin,self.device)

    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        self.initial_img = self.preprocess_input(image)
        self.results = self.net.infer(inputs={self.input_name: self.initial_img})
        self.output = self.preprocess_output(self.results, image)
        left_eye_x_min = self.output['left_eye_x_coordinate'] - 10
        left_eye_x_max = self.output['left_eye_x_coordinate'] + 10
        left_eye_y_min = self.output['left_eye_y_coordinate'] - 10
        left_eye_y_max = self.output['left_eye_y_coordinate'] + 10

        right_eye_x_min = self.output['right_eye_x_coordinate'] - 10
        right_eye_x_max = self.output['right_eye_x_coordinate'] + 10
        right_eye_y_min = self.output['right_eye_y_coordinate'] - 10
        right_eye_y_max = self.output['right_eye_y_coordinate'] + 10

        self.eye_coord = [[left_eye_x_min, left_eye_y_min, left_eye_x_max, left_eye_y_max],
                          [right_eye_x_min, right_eye_y_min, right_eye_x_max, right_eye_y_max]]
        left_eye_image = image[left_eye_x_min:left_eye_x_max, left_eye_y_min:left_eye_y_max]
        right_eye_image = image[right_eye_x_min:right_eye_x_max, right_eye_y_min:right_eye_y_max]

        return left_eye_image, right_eye_image, self.eye_coord

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This method is where you can do that.
        '''

        pre_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        pre_frame = pre_frame.transpose((2, 0, 1))
        pre_frame = pre_frame.reshape(1, *pre_frame.shape)

        return pre_frame

    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''

        outputs = outputs[self.output_name][0]
        left_eye_x_coordinate = int(outputs[0] * image.shape[1])
        left_eye_y_coordinate = int(outputs[1] * image.shape[0])
        right_eye_x_coordinate = int(outputs[2] * image.shape[1])
        right_eye_y_coordinate = int(outputs[3] * image.shape[0])

        return {'left_eye_x_coordinate': left_eye_x_coordinate, 'left_eye_y_coordinate': left_eye_y_coordinate,
                'right_eye_x_coordinate': right_eye_x_coordinate, 'right_eye_y_coordinate': right_eye_y_coordinate}
