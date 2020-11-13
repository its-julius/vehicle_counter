import cv2
import numpy as np
import time


# Prep step: Find out the size of the video file
def find_dimensions(url):
    cap = cv2.VideoCapture(url)
    dimensions = (640, 360)  # width, height
    # Let us first figure out the dimensions of the video
    try:
        result, frame = cap.read()
        if result:
            return frame.shape
        else:
            print("Error in first grab")
            return dimensions
    except Exception as e:
        print(e)


url = 'http://119.2.50.116:83/mjpg/video.mjpg'
framerate = 5
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
dimensions = find_dimensions(url)
writer = cv2.VideoWriter('../output--{0}.mp4'.format(time.strftime('%y-%m-%d-%H-%M')),
                         fourcc, framerate, (dimensions[1], dimensions[0]))

while True:
    try:
        cap = cv2.VideoCapture(url)
        result, frame = cap.read()
        if not result:
            print("Error in cap.read()")  # this is for preventing a breaking error
            # break;
            pass
        print('Capturing ...')
        writer.write(frame)
    except KeyboardInterrupt:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print('Video Stopping ...')
        break
