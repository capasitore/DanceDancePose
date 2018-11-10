#
# This demo shows how to use cvui in multiple windows relying on rows and columns.
#
# Copyright (c) 2018 Fernando Bevilacqua <dovyski@gmail.com>
# Licensed under the MIT license.
#

import numpy as np
import cv2
import cvui
import random

WINDOW1_NAME = 'Window 1'
WINDOW2_NAME = 'Windows 2'


def main():
	# We have one mat for each window.
	frame1 = np.zeros((1024, 768, 3), np.uint8)

	# Create variables used by some components
	window1_values = []
	window2_values = []


	img = cv2.imread('Images/yoga.jpg', cv2.IMREAD_COLOR)
	imgRed = cv2.imread('Images/mic.jpg', cv2.IMREAD_COLOR)
	imgGray = cv2.imread('Images/gamb.jpg', cv2.IMREAD_COLOR)
	img = cv2.resize(img, (200, 200))
	imgRed = cv2.resize(imgGray, (200, 200))
	imgGray = cv2.resize(imgRed, (200, 200))
		
	padding = 10

	# Fill the vector with a few random values
	for i in range(0, 20):
		window1_values.append(random.uniform(0., 300.0))
		window2_values.append(random.uniform(0., 300.0))

	# Start two OpenCV windows
	cv2.namedWindow(WINDOW1_NAME)
	cv2.namedWindow(WINDOW2_NAME)

	# Init cvui and inform it to use the first window as the default one.
	# cvui.init() will automatically watch the informed window.
	cvui.init(WINDOW1_NAME)

	# Tell cvui to keep track of mouse events in window2 as well.
	cvui.watch(WINDOW2_NAME)

	while (True):
		# Inform cvui that all subsequent component calls and events are related to window 1.
		cvui.context(WINDOW1_NAME)

		# Fill the frame with a nice color
		frame1[:] = (49, 52, 49)

		cvui.beginRow(frame1, 10, 20, -1, -1, 10)
		cvui.image(img)
		cvui.button(img, imgGray, imgRed)
		cvui.endRow()

		# Update all components of window1, e.g. mouse clicks, and show it.
		cvui.update(WINDOW1_NAME)
		cv2.imshow(WINDOW1_NAME, frame1)


		# Check if ESC key was pressed
		if cv2.waitKey(20) == 27:
			break


if __name__ == '__main__':
	main()
