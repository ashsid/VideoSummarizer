from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np
import glob
import subprocess
import imutils
from skimage.measure import compare_ssim

origs_imgs_names=[]
unique_chalks=[]

def getFrame(sec,count):
	vidcap = cv2.VideoCapture('condensed.mp4')
	vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
	hasFrames,image = vidcap.read()
	if hasFrames:
		cv2.imwrite(str(count)+".jpg", image)     # save frame as JPG file
	return hasFrames

def framegen():
	cap = cv2.VideoCapture("condensed.mp4")
	fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	duration = frame_count/fps
	mins = int(duration//60)
	print(mins)
	if(mins<=10):
		frameRate=3
	elif(mins>10 and mins<20):
		frameRate=6
	else:
		frameRate=9
	sec = 0
	count=1
	success = getFrame(sec,count)
	while success:
		count = count +1
		sec = sec+frameRate
		sec=round(sec,2)
		success=getFrame(sec,count)

def chalk():
	filenames = [img for img in glob.glob("[0-9]*.jpg")]
	filenames=[s.replace(".jpg","") for s in filenames]
	filenames.sort(key=int)
	filenames=[s+".jpg" for s in filenames]
	print(filenames)
	k=1
	for i in filenames:
		img = cv2.imread(i,0)
		ret,th = cv2.threshold(img,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		#cv2.imwrite("img_thresh.jpg",th)
		thresh = cv2.erode(th, None, iterations=2)
		thresh = cv2.dilate(thresh, None, iterations=4)
		#cv2.imwrite("img_thresh_new.jpg",thresh)
		ch = th - thresh
		name=str(k)+"_chalk.jpg"
		cv2.imwrite(name,ch)
		k=k+1

def delete_img_list(del_list):
	print("------------------REMOVING-------------------")
	for d in del_list:
		print(d)
		os.remove(d)

def delete_img_name_list_origs(del_list):
	print("------------------REMOVING LIST ORIGINAL-------------------")
	for d in del_list:
		print(d)
		origs_imgs_names.remove(d)
def delete_img_name_list_chalks(del_list):
	print("------------------REMOVING LIST CHALKS-------------------")
	for d in del_list:
		print(d)
		unique_chalks.remove(d)

def refine_chalks():
	origs = [ori for ori in glob.glob("[0-9].jpg")]
	origs = origs + [ori for ori in glob.glob("[0-9][0-9].jpg")]
	origs = origs + [ori for ori in glob.glob("[0-9][0-9][0-9].jpg")]
	print(origs)
	origs = [s.replace(".jpg","") for s in origs]
	origs.sort(key=int)
	origs = [s+".jpg" for s in origs]
	print(origs)

	chalks = [img for img in glob.glob("[0-9]*_chalk.jpg")]
	chalks=[s.replace("_chalk.jpg","") for s in chalks]
	chalks.sort(key=int)
	chalks=[s+"_chalk.jpg" for s in chalks]
	print(chalks)
	first=cv2.imread(chalks[0],0)
	#text_cache = pytesseract.image_to_string(first)
	score_int = 1

	for i in range(1,len(chalks)):
		second = cv2.imread(chalks[i],0)
		#text2 = pytesseract.image_to_string(second)
		print("-------------------------------------------")
		print("comparing",chalks[i-1],"and",chalks[i])
		(score, diff) = compare_ssim(first, second, full=True)
		print(abs(score_int - score))
		print("Score",score)
		if(abs(score_int-score) >= 0.06):
			print("Found one unique challboard representation:",chalks[i-1])
			origs_imgs_names.append(origs[i-1])
			unique_chalks.append(chalks[i-1])
		else:
			print("Skipping ....As they are the same")
		first=second
		score_int = score
	delete_img_list([x for x in origs if x not in origs_imgs_names])
	delete_img_list([x for x in chalks if x not in unique_chalks])

	temp=[]
	temp1=[]

	#remove empty ones that do not make sense
	for j in range(len(unique_chalks)):
		ch = cv2.imread(unique_chalks[j],0)
		text = pytesseract.image_to_string(ch)
		if(len(text)<=15):
			temp.append(unique_chalks[j])
			temp1.append(origs_imgs_names[j])

	delete_img_list(temp+temp1)
	delete_img_name_list_chalks(temp)
	delete_img_name_list_origs(temp1)

	temp=[]
	temp1=[]

	length=0
	#keep only unique content
	for j in range(len(unique_chalks)):
		ch = cv2.imread(unique_chalks[j],0)
		text = pytesseract.image_to_string(ch)
		print("for Image ",unique_chalks[j])
		print("-----------------------------")
		print(text)
		print("Length of text")
		print(len(text))
		if(abs(length-len(text))<=15):
			temp.append(unique_chalks[j+1])
			temp1.append(origs_imgs_names[j+1])
		length=len(text)

	delete_img_list(temp+temp1)
	delete_img_name_list_chalks(temp)
	delete_img_name_list_origs(temp1)
	temp=[]
	temp1=[]


	first=cv2.imread(unique_chalks[0],0)
	for i in range(1,len(unique_chalks)):
		second = cv2.imread(unique_chalks[i],0)
		print("-------------------------------------------")
		print("comparing",unique_chalks[i-1],"and",unique_chalks[i])
		(score, diff) = compare_ssim(first, second, full=True)
		print("Score",score)
		if(score >= 0.875):
			print("Found one unique challboard representation:",unique_chalks[i])
			temp1.append(origs_imgs_names[i-1])
			temp.append(unique_chalks[i-1])
		else:
			print("Skipping ....As they are the same")
		first=second

	delete_img_list(temp+temp1)
	delete_img_name_list_chalks(temp)
	delete_img_name_list_origs(temp1)
