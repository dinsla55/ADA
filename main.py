
import numpy as np
import tkinter
import cv2
from matplotlib import pyplot as plt
from unpackSift import unpackSIFTOctave
from matplotlib.mlab import PCA
from eigenVec import getProjectEig
import math
from getPCADesc import getPCADesc
from percent import getPercent


img1 = cv2.imread('img1.ppm',0)
img2 = cv2.imread('img3.ppm',0)         # queryImage


orb = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2= orb.detectAndCompute(img2,None)

#with the calculated SIFT FEATURE WE NOW Calc the PCA SIFT
amountProject=20
A=getProjectEig(amountProject)
print(A.shape)
patchsize=41
desPca1=getPCADesc(img1,kp1,patchsize,A)
desPca2=getPCADesc(img2,kp2,patchsize,A)


bf= cv2.BFMatcher(cv2.NORM_L2,crossCheck=False)

# BFMatcher with default params

matches2 = bf.match(desPca1,desPca2)
matches = bf.match(des1,des2,)

#matchpicture
img3=cv2.drawMatches(img1,kp1,img2,kp2,matches[:35],None,flags=2)


 #Apply ratio test
 ##TODO not important RANSAC
good = []
for i in range(len(matches)):
    good.append(matches[i])


src_pts=np.zeros((len(good),2),dtype=np.float32)
dst_pts=np.zeros((len(good),2),dtype=np.float32)
for i in range(len(good)):
    src_pts[i,:] =  kp1[good[i].queryIdx].pt
    dst_pts [i,]=  kp2[good[i].trainIdx].pt


M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

#p<rint(M)
#print(M[0][1])
percent= getPercent(M,src_pts,dst_pts)


print(percent)




height, width =img2.shape
img4=cv2.warpPerspective(img1,M,(width,height) )

dst=cv2.addWeighted(img2,0.4,img4,0.6,0)


cv2.imshow("rotated pitcture",dst )
#cv2.waitKey(0)
