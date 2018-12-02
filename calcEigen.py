import numpy as np
import cv2
from eigenVec import getProjectEig

from getPCADescPrint import getPCADescPrint


img1 = cv2.imread('img1.ppm',0)



orb = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)


#with the calculated SIFT FEATURE WE NOW Calc the PCA SIFT
amountProject=20
A=getProjectEig(amountProject)
print(A.shape)
patchsize=41
des1=getPCADescPrint(img1,kp1,patchsize)
print(des1)
print(des1.shape)
abc=des1.astype(int)
np.savetxt("testPca.txt",abc,fmt="%i",delimiter=" ",newline="\n")
