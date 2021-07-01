import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork,IECore
import math
import logging
logging.basicConfig(filename='app.log', filemode='w')

def load_model(model_xml,model_bin,device="CPU",extension=None):
    #This method is for loading the model to the device specified by the user.
    #If your model requires any Plugins, this is where you can load them.

    core = IECore()
    model = core.read_network(model=model_xml, weights=model_bin)

    supported_layers = core.query_network(network=model, device_name=device)
    unsupported_layers = [R for R in model.layers.keys() if R not in supported_layers]

    if len(unsupported_layers) != 0:
        print("Unsupported layers found")
        logging.error("Unsupported layer found")

        core.add_extension(extension,device)
        supported_layers = core.query_network(network=model, device_name=device)
        unsupported_layers = [R for R in model.layers.keys() if R not in supported_layers]
    net = core.load_network(network=model, device_name=device, num_requests=1)
    return net