#!/usr/local/bin/python

import os
import cv2

path = '/Users/dgu/Desktop/merl/'

for parent, dirnames, filenames in os.walk(path):
  for filename in filenames:
    if filename.endswith('.jpg'):
      image_name = os.path.join(parent, filename)
      frame = cv2.imread(image_name)
      cv2.resize(frame, (200, 100))
      yA = 160
      yB = 632
      xA = 180
      xB = 910
      # new_image = frame[yA:yB, xA:xB]
      # cv2.imwrite(image_name, new_image)
      cv2.imshow('test', frame)
      cv2.waitKey(0)
