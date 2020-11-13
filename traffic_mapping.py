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


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


cap = cv2.VideoCapture('demo_videos/9.mp4')
r, frame = cap.read()
print('Resolution: ' + str(frame.shape[0]) + ' x ' + str(frame.shape[1]))

while True:
    # Capture frame-by-frame
    time1 = time.time()
    ret, frame = cap.read()
    # frame = rescale_frame(frame, percent=300)
    mapping = np.ones(shape=[frame.shape[0], frame.shape[1], 3], dtype=np.uint8) * 255
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # point = [(430, 450), (535, 430), (1935, 960), (1500, 1350)]
    # point = [(660, 485), (960, 510), (835, 940), (90, 835)]
    # cv2.line(frame, point[0], point[3], (0, 0, 255), 3)
    # cv2.line(frame, point[1], point[2], (0, 0, 255), 3)
    ref_point = np.float32([[660, 485], [960, 510], [90, 835], [835, 940]])
    dst_point = np.float32([[900, 615], [1300, 615], [900, 1065], [1300, 1065]])
    # dst_point = np.float32([[0, 0], [400, 0], [0, 450], [400, 450]])
    matrix = cv2.getPerspectiveTransform(ref_point, dst_point)
    perspective = cv2.warpPerspective(frame, matrix, (1920, 1080))
    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('perspective', perspective)
    # cv2.imshow('prediction', prediction)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    print('Time each frame: ' + str(time.time()-time1) + 'ms')
    print('FPS: ' + str(1/(time.time()-time1)))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()