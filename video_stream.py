import numpy as np
import cv2

cap = cv2.VideoCapture('http://119.2.50.116:83/mjpg/video.mjpg')

r, frame = cap.read()
print('Resolution: ' + str(frame.shape[0]) + ' x ' + str(frame.shape[1]))


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = rescale_frame(frame, percent=300)
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
