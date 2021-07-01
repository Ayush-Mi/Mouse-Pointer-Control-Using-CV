import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork,IECore
import math
import logging
logging.basicConfig(filename='app.log', filemode='w')
from model import load_model

#Head Pose Estimation Model
class HeadPoseEstimationClass:
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
        self.output_list = self.preprocess_output(self.results)
        return self.output_list

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

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        output = []
        output.append(outputs['angle_y_fc'].tolist()[0][0])
        output.append(outputs['angle_p_fc'].tolist()[0][0])
        output.append(outputs['angle_r_fc'].tolist()[0][0])
        return output

    