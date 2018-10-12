import numpy as np
import cv2
import func as f
from matplotlib import pyplot as plt

img1 = cv2.imread('1.jpg',0)          # queryImage
img2 = cv2.imread('2.jpg',0) # trainImage

H=f.findAffineTransform(img1,img2)

#NOT FINISHED!
cameraMatrix= np.zeros((3,3,1),np.uint8)
cameraMatrix[:]=1

(rows, cols) = img1.shape

R= np.zeros((3,3,1),np.uint8)
t= np.zeros((3,1,1),np.uint8)
cv2.decomposeHomographyMat(H,cameraMatrix,R,t)

Rtest= np.zeros((3,3,1),np.float32)
Rtest[:]=1

out=cv2.warpPerspective(img1,Rtest,(cols,rows))
cv2.imshow("Rotated Image",img1)


cv2.waitKey(0)
#print(status)




