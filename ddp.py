import logging
import numpy as np
import time
import cv2
import cvui
import base64

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

Estimator = TfPoseEstimator
WINDOW1_NAME = 'Dance Dance Pose'
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


frame1 = np.zeros((768, 1024, 3), np.uint8)

inferred = infer(posed)
original = cv2.imread(posed)

# timeout_demo = time.time() + 5
# while True:
#     cv2.imshow('original', original)
#     if time.time() > timeout_demo:
#         break

cv2.namedWindow(WINDOW1_NAME)
cvui.init(WINDOW1_NAME)

time.sleep(5)
if original.shape[0] != 480 or original.shape[1] != 640:
      original = cv2.resize(original, (368, 368))
inferred = inferred - original
inferred=cv2.copyMakeBorder(inferred[:,int(np.nonzero(inferred)[1][0]/2):],0,0,0,int(np.nonzero(inferred)[1][0]/2),cv2.BORDER_REPLICATE)
# while True:
#   cv2.imshow('image', inferred)
#   if cv2.waitKey(1) & 0xFF == ord('q'):
#     break
timeout = time.time() + 10
capture = cv2.VideoCapture(0)
counter = [time.time()]
x = 1
while True:
  cvui.context(WINDOW1_NAME)

  ret, frame = capture.read()
  frame1[:] = (49, 52, 49)
  gray = frame

  if gray.shape[0] != 480 or gray.shape[1] != 640:
    gray = cv2.resize(gray, (368, 368))

  dst = cv2.addWeighted(inferred, 0.5, gray, 0.5, 0)

  cvui.beginRow(frame1, 10, 20, -1, -1, 30)

  cvui.image(dst)
  cvui.image(original)
#   cv2.imshow('frame', dst)
  cvui.endRow()

  cvui.beginRow(frame1, 10, 400, -1, -1, 30)
  cvui.counter(frame1, 100, 410, counter, 0.1, '%.1f')
  counter = [timeout - time.time() for x in counter]
  cvui.text(frame1, 10, 410, "Tick tick")
  cvui.endRow()


  cvui.update(WINDOW1_NAME)
  cv2.imshow(WINDOW1_NAME, frame1)
#   time.sleep(5)
  if cv2.waitKey(1) & 0xFF == ord('q') or time.time() > timeout:
#   if cv2.waitKey(1) & 0xFF == ord('q'):
    filename = 'captures/capture' + \
        str(int(x)) + ".png"
    x = x+1
    cv2.imwrite(filename, frame)
    break


inferred_capture = infer(filename)
original_inferred=cv2.imread(filename)
if original_inferred.shape[0] != 4380 or original_inferred.shape[1] != 640:
    inferred_capture = cv2.resize(inferred_capture, (368, 368))
    original_inferred = cv2.resize(original_inferred, (368, 368))
while True:
    final = cv2.addWeighted(inferred, 0.5, inferred_capture, 0.5, 0)
    cv2.imshow('final', final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

diff_inferred=inferred_capture-original_inferred
bw_inferred = cv2.cvtColor(diff_inferred, cv2.COLOR_BGR2GRAY)
bw_inferred[bw_inferred >= 1] = 1
bw_inferred[bw_inferred < 1] = 0

bw_orig_inferred = cv2.cvtColor(inferred, cv2.COLOR_BGR2GRAY)
bw_orig_inferred[bw_orig_inferred >= 1] = 1
bw_orig_inferred[bw_orig_inferred < 1] = 0
total = bw_orig_inferred == bw_inferred
#total=np.count_nonzero(bw_inferred)
#diff_images=bw_orig_inferred-bw_inferred
#diff_images[diff_images>=1]=1
#diff_images[diff_images<1]=0
#reduction=np.count_nonzero(diff_images)

print('')
print('')
print(1-np.sum(total)/np.size(total))

def deleteCaptures():
    pass

# Delete all captures
deleteCaptures()
