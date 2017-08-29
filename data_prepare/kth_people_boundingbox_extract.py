import numpy as np
import cv2
import glob
import os
import sys
from imutils.object_detection import non_max_suppression


def limit(num, minimum=1, maximum=255):
  """Limits input 'num' between minimum and maximum values.
  Default minimum value is 1 and maximum value is 255."""
  return max(min(num, maximum), minimum)

if __name__ == '__main__':

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    filename = "/Users/dgu/Desktop/kth/"
    total = 606
    count = 0.0
    for parent, dirnames, filenames in os.walk(filename):
        # create the new directory
        new_directory = parent.replace("kth", "new_kth")
        os.makedirs(new_directory)
        for filename in filenames:
            if filename.endswith('.jpg'):
                image_name = os.path.join(parent, filename) 
                frame = cv2.imread(image_name)
                frame = cv2.resize(frame, (0,0), fx=1.5, fy=1.5)
                found,w = hog.detectMultiScale(frame, winStride=(4,4), padding=(8,8), scale=1.05)
                rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in found])
                pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
                new_img = None
                for (xA, yA, xB, yB) in pick:
                    # cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
                    xA = limit(xA, 0, 240)
                    xB = limit(xB, 0, 240)
                    yA = limit(yA, 0, 180)
                    yB = limit(yB, 0, 180)
                    # print(xA, yA, xB, yB)
                    new_img = frame[yA:yB, xA:xB] 
                # write new image
                # change folder name
                new_path = image_name.replace("kth", "new_kth")

                # draw_detections(frame,found)
                if new_img is not None:
                    # cv2.imshow('feed',new_img)
                    cv2.imwrite(new_path, new_img)
                # ch = 0xFF & cv2.waitKey(1)
                # if ch == 27:
                #     sys.exit()
        count += 1
        print(count / total)