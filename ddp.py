import cv2, glob, time, random
import numpy as np
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


image_poses = glob.glob('Images/*')

posNo = random.randint(0, len(image_poses) - 1)

capture = cv2.VideoCapture(0)
current_image = cv2.imread(image_poses[posNo], 0)

if current_image.shape[0] != 480 or current_image.shape[1] != 640:
  current_image = cv2.resize(current_image, (640, 480))

timeout = time.time() + 5

while True:
  ret, frame  = capture.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  if gray.shape[0] != 480 or gray.shape[1] != 640:
    gray = cv2.resize(gray, (640, 480))

  dst = cv2.addWeighted(current_image, 0.5, gray, 0.5, 0)

  cv2.imshow('frame', dst)
  if cv2.waitKey(1) & 0xFF == ord('q') or time.time() > timeout:
    break


capture.release()
cv2.destroyAllWindows()


