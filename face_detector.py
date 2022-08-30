# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:52:00 2020

@author: hp
"""

import cv2
import numpy as np
from utils.image_utils import predict, img_preprocess,image_preprocess_person_detect,postprocess_bbbox,postprocess_boxes,nms

import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare


def get_face_detector(modelFile=None,
                      configFile=None,
                      quantized=False):
    """
    Get the face detection caffe model of OpenCV's DNN module
    
    Parameters
    ----------
    modelFile : string, optional
        Path to model file. The default is "models/res10_300x300_ssd_iter_140000.caffemodel" or models/opencv_face_detector_uint8.pb" based on quantization.
    configFile : string, optional
        Path to config file. The default is "models/deploy.prototxt" or "models/opencv_face_detector.pbtxt" based on quantization.
    quantization: bool, optional
        Determines whether to use quantized tf model or unquantized caffe model. The default is False.
    
    Returns
    -------
    model : dnn_Net

    """
    if quantized:
        if modelFile == None:
            modelFile = "utils/models/opencv_face_detector_uint8.pb"
        if configFile == None:
            configFile = "utils/models/opencv_face_detector.pbtxt"
        model = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
        
    else:
        if modelFile == None:
            modelFile = "utils/models/res10_300x300_ssd_iter_140000.caffemodel"
        if configFile == None:
            configFile = "utils/models/deploy.prototxt"
        model = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return model

def get_onnx_face_detector():
    return onnx.load('utils/models/ultra_light_640.onnx')

def find_faces(img, model):
    """
    Find the faces in an image
    
    Parameters
    ----------
    img : np.uint8
        Image to find faces from
    model : dnn_Net
        Face detection model

    Returns
    -------
    faces : list
        List of coordinates of the faces detected in the image

    """
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    res = model.forward()
    faces = []
    for i in range(res.shape[2]):
        confidence = res[0, 0, i, 2]
        if confidence > 0.5:
            box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            faces.append([x, y, x1, y1])
    return faces

def find_faces_onnx(img, model):
    """
    Find the faces in an image
    
    Parameters
    ----------
    img : np.uint8
        Image to find faces from
    model : ultra_light face detector
        Face detection model

    Returns
    -------
    faces : list
        List of coordinates of the faces detected in the image

    """
    predictor = prepare(model)
    ort_session = ort.InferenceSession('utils/models/ultra_light_640.onnx',providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    
    height, width = img.shape[:2]
    img_proc = img_preprocess(img)
    
    confidences, boxes = ort_session.run(None, {input_name: img_proc})
    boxes, labels, probs = predict(width, height, confidences, boxes, 0.7)
    
    if boxes.shape[0] > 1: 
      boxes_list = list(boxes)
      boxes_list.sort(key=lambda x:x[2]-x[0])
   
      if (boxes_list[-1][2]-boxes_list[-1][0])-(boxes_list[-2][2]-boxes_list[-2][0]) >= 60:
        boxes = np.expand_dims(boxes_list[-1],axis=0)
     
    faces = []   
    for i in range(boxes.shape[0]):
      box = boxes[i, :]
      x1, y1, x2, y2 = box
      faces.append([x1, y1, x2, y2])
      
    return faces
    
def find_objects_onnx(original_image, input_size,input_name,output_names,sess,ANCHORS,STRIDES,XYSCALE):
    original_image_size = original_image.shape[:2]

    image_data = image_preprocess_person_detect(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    detections = sess.run(output_names, {input_name: image_data})

    pred_bbox = postprocess_bbbox(detections, ANCHORS, STRIDES, XYSCALE)
    bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.7)
    bboxes = nms(bboxes, 0.213, method='nms')
      
    return bboxes

def draw_faces(img, faces):
    """
    Draw faces on image

    Parameters
    ----------
    img : np.uint8
        Image to draw faces on
    faces : List of face coordinates
        Coordinates of faces to draw

    Returns
    -------
    None.

    """
    for x, y, x1, y1 in faces:
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 3)
        
