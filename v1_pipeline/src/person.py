import os
import time
import itertools
from collections import deque
from typing import Any,List,Tuple, Union
import numpy
import cv2
import torch

from src.person_utils import (
    YoloPostprocessor,load_image,postprocess_nms,

)


_YOLO_CLASSES = [
    "person",
]

_YOLO_CLASS_COLORS = list(itertools.product([0, 255, 128, 64, 192], repeat=3))
_YOLO_CLASS_COLORS.remove((255, 255, 255))  # remove white from possible colors
_YOLO_CLASS_COLORS.remove((0, 0, 0))  # remove black from possible colors

DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"
TORCH_ENGINE = "torch"



def _preprocess_batch(engine,device,quantized_inputs,fp16,batch: numpy.ndarray) -> Union[numpy.ndarray, torch.Tensor]:
    if len(batch.shape) == 3:
        batch = batch.reshape(1, *batch.shape)
    if engine == TORCH_ENGINE:
        batch = torch.from_numpy(batch.copy())
        batch = batch.to(device)
        batch = batch.half() if fp16 else batch.float()
        batch /= 255.0
    else:
        if quantized_inputs == None:
            batch = batch.astype(numpy.float32) / 255.0
        batch = numpy.ascontiguousarray(batch)
    return batch


def _run_model(
    engine, model: Any, batch: Union[numpy.ndarray, torch.Tensor]
) -> List[Union[numpy.ndarray, torch.Tensor]]:
    outputs = None
    if engine == TORCH_ENGINE:
        outputs = model(batch)
    elif engine == ORT_ENGINE:
        outputs = model.run(
            [out.name for out in model.get_outputs()],  # outputs
            {model.get_inputs()[0].name: batch},  # inputs dict
        )
    else:  # deepsparse
        outputs = model.run([batch])
    return outputs




def _annotate_image(
    img: numpy.ndarray,
    outputs: numpy.ndarray,
    score_threshold: float = 0.35,
    model_input_size: Tuple[int, int] = None,
) -> numpy.ndarray:
    """
    Draws bounding boxes on predictions of a detection model

    :param img: Original image to annotate (no pre-processing needed)
    :param outputs: numpy array of nms outputs for the image from postprocess_nms
    :param score_threshold: minimum score a detection should have to be annotated
        on the image. Default is 0.35
    :param model_input_size: 2-tuple of expected input size for the given model to
        be used for bounding box scaling with original image. Scaling will not
        be applied if model_input_size is None. Default is None
    :param images_per_sec: optional images per second to annotate the left corner
        of the image with
    :return: the original image annotated with the given bounding boxes
    """
    img_res = numpy.copy(img)
    img_res2 = numpy.copy(img)

    boxes = outputs[:, 0:4]
    scores = outputs[:, 4]
    labels = outputs[:, 5].astype(int)

    scale_y = img.shape[0] / (1.0 * model_input_size[0]) if model_input_size else 1.0
    scale_x = img.shape[1] / (1.0 * model_input_size[1]) if model_input_size else 1.0

    count=0


    if(len(boxes)>0):

        for idx in range(boxes.shape[0]):
            label = labels[idx].item()



            if scores[idx] > score_threshold:
                count+=1
                annotation_text = (
                    f"{_YOLO_CLASSES[label]}: {scores[idx]:.0%}"

                    if 0 <= label < len(_YOLO_CLASSES)
                    else f"{scores[idx]:.0%}"
                )

        return count,img_res

    else:

        return 0,img_res2







def person_detector(source,image_shape,model,quantized_inputs,fp16,has_postprocessing,conf_thres,engine,device):

    loader = load_image(source,image_shape)

    postprocessor = (
        YoloPostprocessor(image_shape, conf_thres)
        if not has_postprocessing
        else None
    )

    inp=loader[0]
    source_img=loader[1]


    # for iteration, (inp, source_img) in enumerate(loader):
    if device not in ["cpu", None]:
        torch.cuda.synchronize()

    # pre-processing
    batch = _preprocess_batch(engine,device,quantized_inputs,fp16,inp)

    # inference
    outputs = _run_model(engine, model, batch)

    # post-processing
    if postprocessor:
        outputs = postprocessor.pre_nms_postprocess(outputs)
    else:
        outputs = outputs[0]  # post-processed values stored in first output

    # NMS
    outputs = postprocess_nms(outputs)[0]

    if device not in ["cpu", None]:
        torch.cuda.synchronize()

    person_count,annotated_img  = _annotate_image(
        source_img,
        outputs,
        score_threshold=conf_thres,
        model_input_size=image_shape,
    )
    return person_count,annotated_img 
