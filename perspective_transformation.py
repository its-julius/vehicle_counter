import argparse
import os
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import PIL
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized


# cap = cv2.VideoCapture('http://119.2.50.116:83/mjpg/video.mjpg')
cap = cv2.VideoCapture('test_video.mp4')


r, frame = cap.read()
print('Resolution: ' + str(frame.shape[0]) + ' x ' + str(frame.shape[1]))

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).fuse().eval()
# model = model.autoshape()


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def detect(source, confidence_thres=0.4, iou_thres=0.45, weights='yolov5s.pt'):

    # Initialize
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(640, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    img_source = letterbox(source, new_shape=640)[0]
    img_source = img_source[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img_source = np.ascontiguousarray(img_source)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    # for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img_source).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, confidence_thres, iou_thres)
    t2 = time_synchronized()

    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, source)

    print (pred)
    print(type(pred))
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s, im0 = '', source
        s += '%gx%g ' % img.shape[2:]  # print string
        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

        # Print time (inference + NMS)
        print('%sDone. (%.3fs)' % (s, t2 - t1))

        # Stream results
        cv2.imshow('Prediction', im0)
        if cv2.waitKey(1) == ord('q'):  # q to quit
            raise StopIteration
    print('Done. (%.3fs)' % (time.time() - t0))


while True:
    # Capture frame-by-frame
    time1 = time.time()
    ret, frame = cap.read()
    # frame = rescale_frame(frame, percent=300)
    prediction = np.copy(frame)
    # prediction = model(frame)
    # print(prediction)
    detect(prediction)

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    point = [(430, 450), (535, 430), (1935, 960), (1500, 1350)]
    cv2.line(frame, point[0], point[3], (0, 0, 255), 3)
    cv2.line(frame, point[1], point[2], (0, 0, 255), 3)

    # Display the resulting frame
    # cv2.imshow('frame', frame)
    cv2.imshow('prediction', prediction)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print('Time each frame: ' + str(time.time()-time1) + 'ms')
    print('FPS: ' + str(1/(time.time()-time1)))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
