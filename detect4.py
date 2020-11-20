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
from mytracking_sort import MyTrackingSort

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadWebcam
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized, model_info

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from my_function import *
from two_dim_map import TwoDimensionalMap

debug_print = True

# YOLO (80 objects)
object_class = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike',
                4: 'aeroplane', 5: 'bus', 6: 'train', 7: ' truck',
                8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
                11: 'stop sign', 12: 'parking meter'}
# MIO-TCD (11 objects)
# object_class = {0: 'articulated_truck', 1: 'bicycle', 2: 'bus', 3: 'car',
#                 4: 'motorcycle', 5: 'motorized_vehicle', 6: 'non-motorized_vehicle',
#                 7: 'pedestrian', 8: 'pickup_truck',
#                 9: 'single_unit_truck', 10: 'work_van'}

is_deep_sort = True


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
    try:
        out, source, weights, view_img, save_txt, imgsz = \
            opt.save_dir, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

        # ~ Initialize Deep SORT
        # :todo Adjust parameters
        if is_deep_sort:
            deepsort = DeepSort('deep_sort/deep_sort/deep/checkpoint/ckpt.t7',
                                max_dist=0.3, min_confidence=0.3,
                                nms_max_overlap=1.0, max_iou_distance=0.7,
                                max_age=70, n_init=3, nn_budget=100,
                                use_cuda=True)

        # ~ Initialize 2D Map
        # :todo Save reference points in a configuration file
        map2d = TwoDimensionalMap('map_2d')
        map2d.setWH(1920, 1380)
        ref_point = np.float32([[660, 485], [960, 510], [90, 835], [835, 940]])
        dst_point = np.float32([[900, 815], [1300, 815], [900, 1265], [1300, 1265]])
        map2d.setTransformation(ref_point, dst_point)
        map2d.setLine((1146, 521), (1078, 648))
        map2d.setLine((687, 465), (834, 476))
        map2d.setLine((343, 585), (2, 774))

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

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)  # dataset = LoadWebcam(source, img_size=imgsz)  #
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
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            time1 = time.time()

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                       classes=[0, 1, 2, 3, 5, 7], agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                # ~ Update 2D Map Background
                orig_im0 = np.copy(im0)
                # map2d.setWH(im0.shape[1], im0.shape[0])
                map2d.setSource(im0)

                # ~ Window for Info
                info_src = np.ones(shape=(800, 800, 3), dtype=np.uint8) * 255

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

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, conf, *xywh) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line) + '\n') % line)

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    det = det.cpu().data.numpy()  # Convert Torch Tensor in GPU to Numpy Array in CPU
                    # mask = np.zeros(det[:, 5].shape, dtype=np.bool)
                    # for n in list(object_class.keys()):  # Delete row if object is not in object_class keys
                    #     mask |= det[:, 5] == n
                    #     # det = np.delete(det, np.where(det[:, 5] == n)[0], axis=0)
                    # for ind, n in enumerate(mask):
                    #     if not n:
                    #         det = np.delete(det, ind, axis=0)

                    # Deep SORT
                    if is_deep_sort:
                        # Pass detection to Deep Sort
                        bbox = det[:, :4]
                        bbox = xyxy_to_xywh_v2(bbox)
                        confidence = det[:, 4]
                        clss = det[:, 5]
                        number_of_object, outputs = deepsort.update(bbox, confidence, orig_im0, cls=clss,
                                                                    return_number_of_object=True)
                        # Update 2D Map
                        if outputs is not None and len(outputs):
                            map2d.update(outputs[:, :4], outputs[:, 4], outputs[:, 5])

                        # print(outputs)
                        # Draw boxes
                        for output in outputs:
                            cv2.rectangle(im0, (output[0], output[1]), (output[2], output[3]), (255, 0, 255), 5)
                            cv2.putText(im0, "ID: " + str(output[4]) + ' - ' + str(object_class[output[5]]),
                                        (output[0], output[1]), 0, 1e-3 * im0.shape[0], (255, 0, 255), 3)

                # Print time (inference + NMS)
                t3 = time.time()
                if debug_print:
                    print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                if view_img:
                    # ~ Show Info
                    if map2d.cross_counter is not None:
                        info_text = '--- \n' \
                                    '- Line 0 -\n' \
                                    'Person: ' + str(map2d.cross_counter[0, 0]) + '\n'  \
                                    'Car: ' + str(map2d.cross_counter[0, 2]) + '\n'  \
                                    'Bus: ' + str(map2d.cross_counter[0, 5]) + '\n'  \
                                    'Truck: ' + str(map2d.cross_counter[0, 7]) + '\n \n'  \
                                    '- Line 1 -\n' \
                                    'Person: ' + str(map2d.cross_counter[1, 0]) + '\n'  \
                                    'Car: ' + str(map2d.cross_counter[1, 2]) + '\n'  \
                                    'Bus: ' + str(map2d.cross_counter[1, 5]) + '\n'  \
                                    'Truck: ' + str(map2d.cross_counter[1, 7]) + '\n \n'  \
                                    '- Line 2 -\n' \
                                    'Person: ' + str(map2d.cross_counter[2, 0]) + '\n' \
                                    'Car: ' + str(map2d.cross_counter[2, 2]) + '\n'  \
                                    'Bus: ' + str(map2d.cross_counter[2, 5]) + '\n'  \
                                    'Truck: ' + str(map2d.cross_counter[2, 7]) + '\n'  \
                                    '---'
                        for j, text_line in enumerate(info_text.split('\n')):
                            y = 50 + j*40
                            cv2.putText(info_src, text_line, (50, y), 0, 1, (255, 0, 0), 2)
                    cv2.imshow('Info', info_src)
                    cv2.imshow(p, im0)
                    map2d.show()
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        return  # raise StopIteration

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
            if debug_print:
                print('Total Time - FPS: %.5f - %.2f' % (dt, 1 / dt))

        if save_txt or save_img:
            print('Results saved to %s' % Path(out))

        print('Done. (%.3fs)' % (time.time() - t0))

    finally:
        dataset.cap.release()
        cv2.destroyAllWindows()
        print('Properly Closed.')


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
