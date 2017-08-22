import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_green = np.array([40,0,0])
upper_green = np.array([200,255,255])

mask = cv2.inRange(hsv, lower_green, upper_green)

cv2.imshow('mask',image)
cv2.waitKey(0)