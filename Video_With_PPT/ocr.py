'''
pip3 install pillow
pip3 install pytesseract
pip3 install opencv-python
pip3 install imutils
pip3 install scikit-image
'''
from PIL import Image
import numpy as np
import pytesseract
import argparse 
import cv2
import os
import glob
import subprocess
import imutils
from skimage.measure import compare_ssim

filenames = [img for img in glob.glob("[0-9]*_chalk.jpg")]

filenames=[s.replace("_chalk.jpg","") for s in filenames]
#print(filenames)
filenames.sort(key=int)
filenames=[s+"_chalk.jpg" for s in filenames]
print(filenames)
unique_chalks=[]
unique_chalks_imgs=[]
first=cv2.imread(filenames[0],0)
#text_cache = pytesseract.image_to_string(first)


for i in range(1,len(filenames)):
	second = cv2.imread(filenames[i],0)
	#text2 = pytesseract.image_to_string(second)
	print("comparing",filenames[i-1],"and",filenames[i])
	(score, diff) = compare_ssim(first, second, full=True)
	#print("Score",score,"diff",diff)
	if(score>=0.99):
		print("Skipping ....As they are the same")
	else:
		print("Found one unique chalkboard representation:",filenames[i-1])
		unique_chalks.append(filenames[i-1])
		unique_chalks_imgs.append(first)
	#text_cache=text2
	first=second
os.system("rm -rf $(ls | grep '[0-9]*.jpg')")
os.system("rm -rf $(ls | grep '[0-9]_chalk*.jpg')")
print(unique_chalks)
k=1
for j in unique_chalks_imgs:
	cv2.imwrite("IMP_CHALKS"+str(k)+".jpg",j)
	k=k+1


