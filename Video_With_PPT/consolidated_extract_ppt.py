'''
pip3 install opencv-python
pip3 install tinytag
pip3 install pillow
pip3 install pytesseract
pip3 install opencv-python
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
from matplotlib import pyplot as plt
import numpy as np
import glob
import subprocess
import imutils
from skimage.measure import compare_ssim
import sys
# vidcap = cv2.VideoCapture('Networking Basics Explained || What is Networking, Types, Topology, Advantages || CCNA.mp4')
def getFrame(title,sec):

    vidcap = cv2.VideoCapture(title)
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames

# tag = TinyTag.get('Networking Basics Explained || What is Networking, Types, Topology, Advantages || CCNA.mp4')

def extractFrames(videoTitle):
    print('Extracting frames......')
    tag = TinyTag.get(videoTitle)
    print('It is %f seconds long.',tag.duration)
    mins = int(videoTitle.duration//60)
    print(mins)
    if(mins<=10):
    	frameRate=3
    elif(mins>10 and mins<20):
    	frameRate=6
    else:
    	frameRate=9
    sec = 0
    frameRate = 1 #//it will capture image in each 0.5 second
    count=1
    success = getFrame(videoTitle,sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(videoTitle,sec)

def extractPPTFrames():
    print('Extracting PPT frames.....')
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

def selectImportantPPTFrames():
    print('Selecting important frames.....')
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

def main(Title):
    extractFrames(Title)
    extractPPTFrames()
    selectImportantPPTFrames()

if __name__ == "__main__":
    main(sys.argv[0])  
