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

from tinytag import TinyTag
import pytesseract
import argparse
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import glob
from PIL import Image
import subprocess
import imutils
from skimage.measure import compare_ssim

# vidcap = cv2.VideoCapture('condensed.mp4')

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames

    # tag = TinyTag.get('condensed.mp4')
def getFrameFrequency(videoTitle):
    print('It is %f seconds long.',videoTitle.duration)
    mins = int(videoTitle.duration//60)
    print(mins)
    if(mins<=10):
    	frameRate=3
    elif(mins>10 and mins<20):
    	frameRate=6
    else:
    	frameRate=9

    sec = 0

    count=1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)

def getChalk():
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
        thresh = cv2.erode(th, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)
        #cv2.imwrite("img_thresh_new.jpg",thresh)
        ch = th - thresh
        name=str(k)+"_chalk.jpg"
        cv2.imwrite(name,ch)
        k=k+1

 def selectImportantChalkBoardRepresentation():
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
        if(len(text)<=15):
            print("------------------------")
            print("REMOVED",new_chalks[j])
            os.remove(new_chalks[j])


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


def main(Title):
    getFrameFrequency(Title)
    getChalk()
    selectImportantChalkBoardRepresentation()

if __name__ == "__main__":
    main(sys.argv[0])            