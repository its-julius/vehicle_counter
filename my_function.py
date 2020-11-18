import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


def xyxy_to_center_point(xyxy):
    """
    Find center point from X1,Y1,X2,Y2 format
    :param xyxy: [X1,Y1,X2,Y2] top left point and bottom right point
    :return: [X, Y] center point
    """
    if np.array(xyxy).ndim > 1 or len(xyxy) > 4:
        raise ValueError('xyxy format: [x1, y1, x2, y2]')
    x_temp = (xyxy[0] + xyxy[2]) / 2
    y_temp = (xyxy[1] + xyxy[3]) / 2
    return np.array([int(x_temp), int(y_temp)])


def xyxy_to_xywh(xyxy):
    """
    Convert XYXY format (x,y top left and x,y bottom right) to XYWH format (x,y center point and width, height).
    :param xyxy: [X1, Y1, X2, Y2]
    :return: [X, Y, W, H]
    """
    if np.array(xyxy).ndim > 1 or len(xyxy) > 4:
        raise ValueError('xyxy format: [x1, y1, x2, y2]')
    x_temp = (xyxy[0] + xyxy[2]) / 2
    y_temp = (xyxy[1] + xyxy[3]) / 2
    w_temp = abs(xyxy[0] - xyxy[2])
    h_temp = abs(xyxy[1] - xyxy[3])
    return np.array([int(x_temp), int(y_temp), int(w_temp), int(h_temp)])


def xyxy_to_xywh_v2(xyxy):
    """
    Convert XYXY format (x,y top left and x,y bottom right) to XYWH format (x,y center point and width, height).
    :param xyxy: [X1, Y1, X2, Y2]
    :return: [X, Y, W, H]
    """
    if isinstance(xyxy, np.ndarray):
        xywh = xyxy.copy()
    elif isinstance(xyxy, torch.Tensor):
        xywh = xyxy.clone()
    else:
        raise ValueError('xyxy type unknown.')
    xywh[:, 2] = abs(xyxy[:, 2] - xyxy[:, 0])
    xywh[:, 3] = abs(xyxy[:, 1] - xyxy[:, 3])
    xywh[:, 0] = xywh[:, 0] + xywh[:, 2]/2
    xywh[:, 1] = xywh[:, 1] + xywh[:, 3]/2
    return xywh


def xywh_to_xyxy(xywh):
    """
    Convert XYWH format (x,y center point and width, height) to XYXY format (x,y top left and x,y bottom right).
    :param xywh: [X, Y, W, H]
    :return: [X1, Y1, X2, Y2]
    """
    if np.array(xywh).ndim > 1 or len(xywh) > 4:
        raise ValueError('xywh format: [x1, y1, width, height]')
    x1 = xywh[0] - xywh[2] / 2
    y1 = xywh[1] - xywh[3] / 2
    x2 = xywh[0] + xywh[2] / 2
    y2 = xywh[1] + xywh[3] / 2
    return np.array([int(x1), int(y1), int(x2), int(y2)])


def point_perspective_transform(point, matrix):
    """
    Perspective transform a point given the transformation matrix
    :param point: [X, Y] point to be transformed
    :param matrix: 3x3 transformation matrix
    :return: [X, Y] transformed point
    """
    matrix = np.asanyarray(matrix)
    if matrix.shape != (3, 3):
        raise ValueError('matrix shape should be (3, 3)')
    x_temp = (matrix[0, 0] * point[0] + matrix[0, 1] * point[1] + matrix[0, 2]) / (
            matrix[2, 0] * point[0] + matrix[2, 1] * point[1] + matrix[2, 2])
    y_temp = (matrix[1, 0] * point[0] + matrix[1, 1] * point[1] + matrix[1, 2]) / (
            matrix[2, 0] * point[0] + matrix[2, 1] * point[1] + matrix[2, 2])
    return np.array([int(x_temp), int(y_temp)])


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)