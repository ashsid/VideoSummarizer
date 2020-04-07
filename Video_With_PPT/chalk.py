'''
pip3 install pillow
pip3 install pytesseract
pip3 install opencv-python
'''
from PIL import Image
import pytesseract
import argparse
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import glob

filenames = [img for img in glob.glob("[0-9]*.jpg")]
filenames=[s.replace(".jpg","") for s in filenames]
#print(filenames)
filenames.sort(key=int)
filenames=[s+".jpg" for s in filenames]
print(filenames)

k=1
for i in filenames:
	img = cv2.imread(i,0)
	ret,th = cv2.threshold(img,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#cv2.imwrite("img_thresh.jpg",th)
	#Next two lines are necessary if lecturer covers green board to extract chalkboard info#
	#---thresh = cv2.erode(th, None, iterations=2)---#
	#---thresh = cv2.dilate(thresh, None, iterations=4)---#
	#cv2.imwrite("img_thresh_new.jpg",thresh)
	#Next line also required
	#--ch = th - thresh--#
	name=str(k)+"_chalk.jpg"
	cv2.imwrite(name,th)
	k=k+1


