import cv2
import numpy as np
import argparse


def rescale_frame(frame, scale=2):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def draw_point(frame, p, color=(255, 255, 0)):
    cv2.circle(frame, p, 5, color, -1)
    return frame


def mouse_click(event, x, y, flag, param):
    global point, point_flag, remove_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        # print('LB')
        point = (x, y)
        point_flag = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        # print('RB')
        point = (x, y)
        remove_flag = True


def show():
    global point, point_flag, remove_flag
    point_list = set()
    src, rescale = opt.source, opt.rescale
    # stream = src.isnumeric() or src.startswith(('rtsp://', 'rtmp://', 'http://')) or src.endswith('.txt')
    try:
        cap = cv2.VideoCapture(int(src)) if src.isnumeric() else cv2.VideoCapture(src)
        ret, frame = cap.read()
        cv2.namedWindow("Source")
        cv2.setMouseCallback("Source", mouse_click)
        print('Resolution: ' + str(frame.shape[0]) + 'x' + str(frame.shape[1]))
        if rescale != 1:
            print('Rescale: ' + str(rescale) + 'x')
        print('FPS: ' + str(cap.get(cv2.CAP_PROP_FPS)))
        if src.isnumeric():
            ts = 1
        else:
            ts = int(1000 / cap.get(cv2.CAP_PROP_FPS))
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Done!')
                break
            if rescale != 1:
                frame = rescale_frame(frame, rescale)
            original_frame = np.copy(frame)
            for p in point_list:
                frame = draw_point(frame, p)
            cv2.imshow('Source', frame)

            key = cv2.waitKey(ts)
            if key == 27 or key == ord('q'):
                exit()
            elif key == 32:
                while True:
                    if point_flag and point is not None:
                        point_list.add(point)
                        print('Point: ' + str(tuple(int(t/rescale) for t in point)))
                        frame = draw_point(frame, point)
                        cv2.imshow('Source', frame)
                        cv2.waitKey(1)
                        point = None
                        point_flag = False
                    elif remove_flag and point is not None:
                        for p in point_list:
                            if abs(p[0]-point[0]) < 10 and abs(p[1]-point[1]) < 10:
                                point_list.remove(p)
                                break
                        frame = np.copy(original_frame)
                        for p in point_list:
                            frame = draw_point(frame, p)
                        cv2.imshow('Source', frame)
                        cv2.waitKey(1)
                        point = None
                        remove_flag = False
                    key = cv2.waitKey(ts)
                    if key == 32:
                        break
                    elif key == 27 or key == ord('q'):
                        exit()
            point = None
            point_flag = False
            remove_flag = False

    finally:
        if len(point_list) == 4:
            print('Result: ')
            for p in point_list:
                print(tuple(int(t/rescale) for t in p))
        print('Exiting...')
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=0, help='Video or Image Source')
    parser.add_argument('--rescale', type=float, default=1, help='Rescale Video or Image')
    opt = parser.parse_args()
    print(opt)
    point = None
    point_flag = False
    remove_flag = False
    show()


'''
How to use (video source):
    1. Run 'python point_capture.py --source /video/path.ext' (--rescale if needed)
    2. Hit SPACEBAR to pause or resume the video
    3. Left click when the video pauses to mark the reference point
    4. Right click to remove the reference point
    5. Press ESC to exit the program
'''