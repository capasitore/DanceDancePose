import logging

import time
import cv2
import base64

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

Estimator = TfPoseEstimator

posed = 'Images/yoga.jpg'


def infer(image, model='mobilenet_thin', resize='368x368', resize_out_ratio=4.0):
    """

    :param image:
    :param model:
    :param resize:
    :param resize_out_ratio:
    :return:
    """
    w, h = model_wh(resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    # estimate human poses from a single image !
    image = common.read_imgfile(image, None, None)
    if image.shape[0] != 480 or image.shape[1] != 640:
      image = cv2.resize(image, (368, 368))
    if image is None:
        raise Exception('Image can not be read, path=%s' % image)
    humans = e.inference(image, resize_to_default=(
        w > 0 and h > 0), upsample_size=resize_out_ratio)

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    return image


inferred = infer(posed)
original = cv2.imread(posed)

timeout_demo = time.time() + 2
while True:
    cv2.imshow('original', original)
    if time.time() > timeout_demo:
        break

if original.shape[0] != 480 or original.shape[1] != 640:
      original = cv2.resize(original, (368, 368))
inferred = inferred - original
# while True:
#   cv2.imshow('image', inferred)
#   if cv2.waitKey(1) & 0xFF == ord('q'):
#     break
timeout = time.time() + 10
capture = cv2.VideoCapture(0)
x = 1
while True:
  ret, frame = capture.read()
  
  gray = frame

  if gray.shape[0] != 480 or gray.shape[1] != 640:
    gray = cv2.resize(gray, (368, 368))

  dst = cv2.addWeighted(inferred, 0.5, gray, 0.5, 0)

  cv2.imshow('frame', dst)
#   time.sleep(5)
  if cv2.waitKey(1) & 0xFF == ord('q') or time.time() > timeout:
#   if cv2.waitKey(1) & 0xFF == ord('q'):
    filename = 'captures/capture' + \
        str(int(x)) + ".png"
    x = x+1
    cv2.imwrite(filename, frame)
    break


inferred_capture = infer(filename)
while True:
    final = cv2.addWeighted(inferred, 0.5, inferred_capture, 0.5, 0)
    cv2.imshow('final', final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

def deleteCaptures():
    pass

# Delete all captures
deleteCaptures()
