import pytesseract
import argparse
import cv2
import os
import numpy as np
import glob
from PIL import Image
import glob
import subprocess
import imutils
from skimage.measure import compare_ssim


def ppt():
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
		name=str(k)+"_chalk.jpg"
		cv2.imwrite(name,th)
		k=k+1
	
def refine_ppt():
	origs = [ori for ori in glob.glob("[0-9].jpg")]
	origs = origs + [ori for ori in glob.glob("[0-9][0-9].jpg")]
	origs = origs + [ori for ori in glob.glob("[0-9][0-9][0-9].jpg")]
	origs = [s.replace(".jpg","") for s in origs]
	origs.sort(key=int)
	origs = [s+".jpg" for s in origs]
	print(origs)


	filenames = [img for img in glob.glob("[0-9]*_chalk.jpg")]

	filenames=[s.replace("_chalk.jpg","") for s in filenames]
	#print(filenames)
	filenames.sort(key=int)
	filenames=[s+"_chalk.jpg" for s in filenames]
	print(filenames)
	unique_chalks=[]
	unique_chalks_imgs=[]
	origs_names=[]
	origs_imgs=[]
	first=cv2.imread(filenames[0],0)
	first_orig=cv2.imread(origs[0])
	#text_cache = pytesseract.image_to_string(first)


	for i in range(1,len(filenames)):
		second = cv2.imread(filenames[i],0)
		second_orig = cv2.imread(origs[i])
		#text2 = pytesseract.image_to_string(second)
		print("comparing",filenames[i-1],"and",filenames[i])
		(score, diff) = compare_ssim(first, second, full=True)
		#print("Score",score,"diff",diff)
		if(score>=0.90):
			print("Skipping ....As they are the same")
		else:
			print("Found one unique chalkboard representation:",filenames[i-1])
			unique_chalks.append(filenames[i-1])
			unique_chalks_imgs.append(first)
			origs_imgs.append(first_orig)
			origs_names.append(origs[i-1])
		#text_cache=text2
		first=second
		first_orig=second_orig
	os.system("rm -rf $(ls | grep '[0-9]*.jpg')")
	os.system("rm -rf $(ls | grep '[0-9]_chalk*.jpg')")
	print(unique_chalks)
	k=1
	for j in range(len(unique_chalks_imgs)):
		cv2.imwrite(str(k)+"_chalk.jpg",unique_chalks_imgs[j])
		cv2.imwrite(str(k)+".jpg",origs_imgs[j])
		k=k+1