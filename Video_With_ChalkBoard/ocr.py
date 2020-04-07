'''
pip3 install pillow
pip3 install pytesseract
pip3 install opencv-python
pip3 install imutils
pip3 install scikit-image
'''

from PIL import Image
import pytesseract
import argparse 
import cv2
import os
import glob
import subprocess
import imutils
from skimage.measure import compare_ssim
import numpy as np

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
score_int = 1
unique_chalks.append(filenames[1])
unique_chalks_imgs.append(first)

for i in range(1,len(filenames)):
	second = cv2.imread(filenames[i],0)
	#text2 = pytesseract.image_to_string(second)
	print("-------------------------------------------")
	print("comparing",filenames[i-1],"and",filenames[i])
	(score, diff) = compare_ssim(first, second, full=True)
	print(abs(score_int - score))
	print("Score",score)
	if(abs(score_int-score) >= 0.06):
		print("Found one unique challboard representation:",filenames[i-1])
		unique_chalks.append(filenames[i-1])
		unique_chalks_imgs.append(first)
	else:
		print("Skipping ....As they are the same")
	first=second
	score_int = score
os.system("rm -rf $(ls | grep '[0-9]*.jpg')")
os.system("rm -rf $(ls | grep '[0-9]_chalk*.jpg')")
print(unique_chalks)
print("fxggxgdf")
k=1
for j in unique_chalks_imgs:
	cv2.imwrite(str(k)+"_IMP_CHALKS.jpg",j)
	k=k+1

new_chalks = [img for img in glob.glob("*.jpg")]
new_chalks = [s.replace("_IMP_CHALKS.jpg","") for s in new_chalks]
new_chalks.sort(key=int)
new_chalks = [s+"_IMP_CHALKS.jpg" for s in new_chalks]
print(new_chalks)
length=0
#remove empty ones that do not make sense
for j in range(len(new_chalks)):
	ch = cv2.imread(new_chalks[j],0)
	text = pytesseract.image_to_string(ch)
	# if(len(text)<=15):
		# print("------------------------")
		# print("REMOVED",new_chalks[j])
		# os.remove(new_chalks[j])


new_chalks = [img for img in glob.glob("*.jpg")]
new_chalks = [s.replace("_IMP_CHALKS.jpg","") for s in new_chalks]
new_chalks.sort(key=int)
new_chalks = [s+"_IMP_CHALKS.jpg" for s in new_chalks]
imp_chalks_imgs=[]
imp_chalks_imgs_names=[]
length=0
#keep only unique content
for j in range(len(new_chalks)):
	ch = cv2.imread(new_chalks[j],0)
	text = pytesseract.image_to_string(ch)
	print("for Image ",new_chalks[j])
	print("-----------------------------")
	print(text)
	print("Length of text")
	print(len(text))
	if(abs(length-len(text))<=15):
		imp_chalks_imgs.append(ch)
		imp_chalks_imgs_names.append(new_chalks[j])
	length=len(text)


first=cv2.imread(new_chalks[0],0)
for i in range(1,len(new_chalks)):
	second = cv2.imread(new_chalks[i],0)
	print("-------------------------------------------")
	print("comparing",new_chalks[i-1],"and",new_chalks[i])
	(score, diff) = compare_ssim(first, second, full=True)
	print("Score",score)
	if(score >= 0.875):
		print("Found one unique challboard representation:",new_chalks[i])
		imp_chalks_imgs.append(second)
		imp_chalks_imgs_names.append(new_chalks[i])
	

	else:
		print("Skipping ....As they are the same")
	first=second

print(imp_chalks_imgs_names)
print(len(imp_chalks_imgs_names))

os.system("rm -rf $(ls | grep '[0-9]_IMP_CHALKS*.jpg')")
k=1
for j in imp_chalks_imgs:
	cv2.imwrite(str(k)+"_IMP_CHALKS.jpg",j)
	k=k+1