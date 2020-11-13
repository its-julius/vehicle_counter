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
from my_function import *

vehicles = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
vehicle_color = {'person': (255, 0, 0), 'bicycle': (0, 255, 0), 'car': (0, 0, 255),
                 'motorcycle': (255, 255, 0), 'bus': (255, 0, 255), 'truck': (0, 255, 255)}


# def bbox_rel(*xyxy):
#     """
#     Calculates the relative bounding box from absolute pixel values.
#     """
#     bbox_left = min([xyxy[0].item(), xyxy[2].item()])
#     bbox_top = min([xyxy[1].item(), xyxy[3].item()])
#     bbox_w = abs(xyxy[0].item() - xyxy[2].item())
#     bbox_h = abs(xyxy[1].item() - xyxy[3].item())
#     x_c = (bbox_left + bbox_w / 2)
#     y_c = (bbox_top + bbox_h / 2)
#     return x_c, y_c, bbox_w, bbox_h


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        id = int(identities[i]) if identities is not None else 0
        palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        color = tuple([int((p * (id ** 2 - id + 1)) % 255) for p in palette])
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.save_dir, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize Deep SORT
    cfg = get_config()
    cfg.merge_from_file('deep_sort/configs/deep_sort.yaml')
    deepsort = DeepSort('deep_sort/deep_sort/deep/checkpoint/ckpt.t7',
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=70, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    past_box = {}

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):  # output dir
        shutil.rmtree(out)  # delete dir
    os.makedirs(out)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:

        time1 = time.time()

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            # Record each center point of detections
            cp_det = []
            map_2d = np.ones(shape=(1280, 1920, 3), dtype=np.uint8) * 255
            map_tracking = np.ones(shape=(1280, 1920, 3), dtype=np.uint8) * 255

            # Calculate transformation matrix
            ref_point = np.float32([[660, 485], [960, 510], [90, 835], [835, 940]])
            dst_point = np.float32([[900, 715], [1300, 715], [900, 1165], [1300, 1165]])
            # dst_point = np.float32([[0, 0], [400, 0], [0, 450], [400, 450]])
            matrix = cv2.getPerspectiveTransform(ref_point, dst_point)

            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Convert X,Y X,Y format to X,Y W,H
                    xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4))
                    # Append center point to cp_det
                    cp_det.append([int(xywh[0, 0]), int(xywh[0, 1])])

                    if names[int(cls)] in vehicles:
                        # Adapt detections to Deep SORT input format
                        # img_h, img_w, _ = im0.shape
                        # x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                        # obj = [x_c, y_c, bbox_w, bbox_h]
                        xyxy_value = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]
                        xywh_value = xyxy_to_xywh(xyxy_value)
                        bbox_xywh.append(xywh_value)
                        confs.append([conf.item()])

                        # my code
                        # Find the center point
                        xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4))
                        cp = [int(xywh[0, 0]), int(xywh[0, 1])]
                        new_cp = point_perspective_transform(cp, matrix)
                        # Plot the center point based
                        # cv2.circle(map_2d, tuple(new_cp), 5, vehicle_color[names[int(cls)]], 5)
                        cv2.circle(map_2d, tuple(new_cp), 5, (0, 255, 255), 5)

                        # my code - end

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, conf, *xywh) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line) + '\n') % line)

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                present_box = {}

                # Deep SORT
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                # Pass detection to Deep Sort
                outputs = deepsort.update(xywhs, confss, im0)
                # print(outputs)
                # Draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    for j, box in enumerate(bbox_xyxy):
                        id_temp = identities[j]
                        temp = xyxy_to_xywh(box)
                        present_box[id_temp] = [temp[0], temp[1]]
                    draw_boxes(im0, bbox_xyxy, identities)
                    for key in present_box:
                        cv2.circle(map_tracking, tuple(present_box[key]), 3, (0, 255, 0), 3)
                    for key in present_box:
                        if key in past_box:
                            # Draw Line?
                            cv2.line(map_tracking, tuple(past_box[key]), tuple(present_box[key]), (0, 0, 255), 3)

                past_box = present_box

                # Plot center points
                # cp_det = np.asanyarray(cp_det)
                # for cp in cp_det:
                #     cv2.circle(im0, tuple(cp), 5, (0,0,255), 5)

                # # Perpective transform center points
                # cp_transform = []
                # for cp in cp_det:
                #     # Manually perspective transform a point
                #     x_temp = (matrix[0,0]*cp[0] + matrix[0,1]*cp[1] + matrix[0,2]) / (matrix[2,0]*cp[0] + matrix[2,1]*cp[1] + matrix[2,2])
                #     y_temp = (matrix[1,0]*cp[0] + matrix[1,1]*cp[1] + matrix[1,2]) / (matrix[2,0]*cp[0] + matrix[2,1]*cp[1] + matrix[2,2])
                #     # point = np.array([cp[0],cp[1],1])
                #     # result = np.matmul(point, matrix)
                #     cp_transform.append([int(x_temp), int(y_temp)])
                #     # cp_transform.append(cv2.perspectiveTransform(np.float32(cp), matrix))
                # print(cp_transform)

            # Print time (inference + NMS)
            t3 = time.time()
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Perspective Transform
            # ref_point = np.float32([[660, 485], [960, 510], [90, 835], [835, 940]])
            # dst_point = np.float32([[900, 615], [1300, 615], [900, 1065], [1300, 1065]])
            # dst_point = np.float32([[0, 0], [400, 0], [0, 450], [400, 450]])
            # matrix = cv2.getPerspectiveTransform(ref_point, dst_point)
            perspective = cv2.warpPerspective(im0, matrix, (1920, 1280))
            # Plot transformed center points
            # cp_det = np.asanyarray(cp_det)
            if 'cp_transform' in locals():
                for cp in cp_transform:
                    cv2.circle(perspective, tuple(cp), 5, (0, 0, 255), 5)

            # Stream results
            if view_img:
                cv2.imshow('perspective', perspective)
                cv2.imshow(p, im0)
                cv2.imshow('2D Map', map_2d)
                cv2.imshow('Tracking', map_tracking)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

        dt = time.time() - time1
        print('Total Time - FPS: %.5f - %.2f' % (dt, 1 / dt))

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
