'''
pip3 install opencv-python
pip3 install tinytag
'''
import cv2
from tinytag import TinyTag
vidcap = cv2.VideoCapture('video.mp4')
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames
tag = TinyTag.get('video.mp4')

print('It is %f seconds long.',tag.duration)
mins = int(tag.duration//60)
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
