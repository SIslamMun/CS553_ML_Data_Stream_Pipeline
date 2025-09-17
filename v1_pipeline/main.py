#import libraries
import cv2
import datetime
import time
import os
import onnx
import sys
import onnxruntime
from redis import Redis
import pickle
import torch
from collections import deque
from typing import Any, List, Union
from src.pipeline import Non_Productive_Time
from deepsparse import compile_model
from src.person_utils import (
    download_pytorch_model_if_stub,
    modify_yolo_onnx_input_shape,
    yolo_onnx_has_postprocessing,
)
from sparseml.onnx.utils import override_model_batch_size



import argparse
import logging
import sys
import cv2
import numpy as np
import base64
from pyflink.common import WatermarkStrategy, Encoder, Types
from pyflink.datastream import StreamExecutionEnvironment, RuntimeExecutionMode
from pyflink.datastream.connectors.file_system import FileSource, StreamFormat, FileSink, OutputFileConfig, RollingPolicy



import logging
from logging.handlers import RotatingFileHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logHandler = RotatingFileHandler('./main.log', backupCount=3)
logHandler.setLevel(logging.INFO)
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)


# initializing all framework engine
DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"
TORCH_ENGINE = "torch"

# REDIS_CLIENT = Redis(host='localhost', port=6379, db=0)

# cam_queue = sys.argv[1]


# loading all types of model
def _load_model(model_filepath,engine,device,fp16,quantized_inputs,num_cores,image_shape):
    """
    LOADING_MODEL
    Args:
        model_filepath: ml model path .onnx or .pt format
        engine: engine framework (deepsparse,torch or onnxruntime).
        device: processing machine (cuda for gpu, cpu only).
        fp16: boolean Floating point value.
        quantized_inputs: quantization value (true,false or none).
        num_cores: total number of cores 
        image_shape: model input shape

    Returns: 
        model: compiled engine model.
        has_postprocessing: boolean result

    """

    # validation
    if device not in [None, "cpu"] and engine != TORCH_ENGINE:
        raise ValueError(f"device {device} is not supported for {engine}")
    if fp16 and engine != TORCH_ENGINE:
        raise ValueError(f"half precision is not supported for {engine}")
    if quantized_inputs and engine == TORCH_ENGINE:
        raise ValueError(f"quantized inputs not supported for {engine}")
    if num_cores is not None and engine == TORCH_ENGINE:
        raise ValueError(
            f"overriding default num_cores not supported for {engine}"
        )
    if (
        num_cores is not None
        and engine == ORT_ENGINE
        and onnxruntime.__version__ < "1.7"
    ):
        raise ValueError(
            "overriding default num_cores not supported for onnxruntime < 1.7.0. "
            "If using an older build with OpenMP, try setting the OMP_NUM_THREADS "
            "environment variable"
        )

    # scale static ONNX graph to desired image shape
    if engine in [DEEPSPARSE_ENGINE, ORT_ENGINE]:
        model_filepath, _ = modify_yolo_onnx_input_shape(
            model_filepath, image_shape
        )
        has_postprocessing = yolo_onnx_has_postprocessing(model_filepath)

    # load model
    if engine == DEEPSPARSE_ENGINE:

        model = compile_model(model_filepath, 1, 4)

    elif engine == ORT_ENGINE:


        sess_options = onnxruntime.SessionOptions()
        if num_cores is not None:
            sess_options.intra_op_num_threads = num_cores
        sess_options.log_severity_level = 3
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        onnx_model = onnx.load(model_filepath)
        override_model_batch_size(onnx_model, 1)
        model = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), sess_options=sess_options
        )

    elif engine == TORCH_ENGINE:
        model_filepath = download_pytorch_model_if_stub(model_filepath)
        model = torch.load(model_filepath)
        if isinstance(model, dict):
            model = model["model"]
        model.to(device)
        model.eval()
        if fp16:

            model.half()
        else:

            model.float()
        has_postprocessing = True

    return model, has_postprocessing

def convert_frame_to_base64(frame):
    ret, buffer = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    # print("base64")
    return base64_image

def decode_base64_to_frame(base64_image):
    base64_image = base64_image.encode('utf-8')
    image_data = np.frombuffer(base64.b64decode(base64_image), np.uint8)
    frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return frame

# class NonProductiveTime():

#     def __init__(self):

#         try:
device = "cpu"


#initializing pipeline object
npt = Non_Productive_Time()
#initializing input shape
image_shape = (416, 416)

if device == 'cpu':

    engine = 'deepsparse'
    #initializing directory path
    person_model_filepath = "deepsparse_models/person_m_fusion.onnx"

    #initializing model input
    p_conf_thresh = 0.35


    pfp16 = False


    num_cores = 4
    
    p_quantized_inputs = True



elif device == "gpu":

    engine = 'torch'
    device == "cuda"

    #initializing directory path
    person_model_filepath = "deepsparse_models/person_m_fusion.pt"

    #initializing model input
    p_conf_thresh = 0.35


    pfp16 = False


    num_cores = None
    
    p_quantized_inputs = False



else:
    print()
    print("________________________________________________________")
    print("Unrecognized Processing Unit. use command 'gpu' or 'cpu'")
    print("________________________________________________________")
    print()
    exit()

person_model, person_has_postprocessing = _load_model(person_model_filepath,engine,device,
pfp16,p_quantized_inputs,num_cores,image_shape)

        # logger.info("person and activity network loaded")

    # except Exception as e:
    #     _, _, exception_traceback = sys.exc_info()
    #     filename = exception_traceback.tb_frame.f_code.co_filename
    #     line_number = exception_traceback.tb_lineno
    #     logger.error('{}'.format(e)+' on line {}'.format(line_number)+' from ' +'{}'.format(filename))


def run(img):
    """
    Run the pipeline.

    """
    # video_path = 'single_person.mp4'
    # cap = cv2.VideoCapture(video_path)
    # while True:
    # frame = img

    # try:

    #     frame = img


    frame = decode_base64_to_frame(img)
    #     # st=time.time()

    input_image = frame
    #     #     #read date and time
    today = datetime.datetime.now()
            #date and time in string format
    date_time = today.strftime("%Y-%m-%d %H:%M:%S")
    #     # print("before pipeline:")
            
    #     # # pipeline call and final output 
    absent_time_calc = npt.pipeline(input_image, date_time,image_shape,person_model,
        p_quantized_inputs,pfp16,person_has_postprocessing,p_conf_thresh,engine,device)


    #     et = time.time()
    #     print("detection_time:",et-st)
    #     fps = 1/(et-st)
    #     fps = int(fps)
    #     fps = str(fps)
    #     print("fps:",fps)
        
    #     print("...........................")

    # except Exception as e:
    #     # print(e)
        
    #     exception_type, exception_object, exception_traceback = sys.exc_info()
    #     filename = exception_traceback.tb_frame.f_code.co_filename
    #     line_number = exception_traceback.tb_lineno
    #     logger.error('{}'.format(e)+' on line {}'.format(line_number)+' from ' +'{}'.format(filename))
    #     # continue
    return input_image




def flink_streaming_engine(input_path):
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_runtime_mode(RuntimeExecutionMode.BATCH)
    # write all the data to one file
    

    env.set_parallelism(1)

    


    ds = env.from_collection(input_path)

    # print(ds)

    
    # compute word count
    ds = ds.flat_map(run)

    # define the sink

    print("Printing result to stdout: ")
    ds.print()

    # submit for execution
    env.execute()
if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        dest='input',
        required=False,
        help='Input file to process.')
    parser.add_argument(
        '--output',
        dest='output',
        required=False,
        help='Output file to write results to.')

    argv = sys.argv[1:]
    known_args, _ = parser.parse_known_args(argv)

    data=[]
    video_path = 'test.png'
    # cap = cv2.VideoCapture(video_path)
    # while cap.isOpened():
    frame = cv2.imread(video_path)
    base_64 = convert_frame_to_base64(frame)
    data.append(""+ base_64 + "")

    # print(word_count_data)
    flink_streaming_engine(data)

