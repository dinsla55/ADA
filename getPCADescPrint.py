
import numpy as np

import cv2
import math
from unpackSift import unpackSIFTOctave

from getPatch import getPatch

def getPCADescPrint(img1,kp1,patchsize):
	counter =0
	des1 = np.empty(shape=(39*39*2),dtype=np.float32)
	file =open("testPca.txt","w")
	patch= np.empty(shape=(patchsize,patchsize),dtype=np.float32)
	rows, cols = img1.shape
	for x in range(counter,len(kp1)):

		oct , layer, scale=unpackSIFTOctave(kp1[x])
		patchsizeQ1=int(patchsize*math.sqrt(2)/2*scale)

		if(kp1[x].pt[0]>(patchsizeQ1) and kp1[x].pt[1]>(patchsizeQ1)
			and kp1[x].pt[0]<(rows-(patchsizeQ1)) and kp1[x].pt[1]<(cols-(patchsizeQ1))):

			patch = getPatch(kp1[x],img1,patchsize)
			counter=x+1

		elif(x==(len(kp1)-1)):
			break
		else:
			continue

		kx = cv2.Sobel(patch,cv2.CV_32F,1,0)
		ky = cv2.Sobel(patch,cv2.CV_32F,0,1)
		#print(kx.shape)
		rowsK,colsK =kx.shape
		kxBoarder=kx[1:rowsK-1,1:colsK-1]
		kyBoarder=ky[1:rowsK-1,1:colsK-1]
		descriptor = np.empty(shape=(kx.size+ky.size),dtype=np.float32)
		descriptor = np.append(kxBoarder,kyBoarder)
		#descProject=np.empty(amountNumber,dtype=np.float32);
		#np.savetxt("testPca.txt",descriptor,delimiter=" ", fmt="%.5e")

		#print(des1.shape)
		#print(descriptor.shape)
		des1=np.vstack([des1,descriptor])

		if(x==(len(kp1)-1)):
			break

	return des1
