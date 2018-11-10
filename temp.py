import numpy as np
import cv2
import time
cap = cv2.VideoCapture(0)
framerate = cap.get(5)
x = 1

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here

    filename = 'captures/capture' + \
        str(int(x)) + ".png"
    x = x+1
    cv2.imwrite(filename, frame)
    time.sleep(5)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
