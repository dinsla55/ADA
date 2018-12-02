
import numpy as np

import cv2
import math
from unpackSift import unpackSIFTOctave

from getPatch import getPatch

def getPCADesc(img1,kp1,patchsize,A):
	counter =0
	amountNumber,tmp=A.shape
	file =open("testPca.txt","w")
	patch= np.empty(shape=(patchsize,patchsize),dtype=np.float32)
	rows, cols = img1.shape
	des1=np.empty(amountNumber,dtype=np.float32);

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

		rowsK,colsK =kx.shape
		kxBoarder=kx[1:rowsK-1,1:colsK-1]
		kyBoarder=ky[1:rowsK-1,1:colsK-1]
		descriptor = np.empty(shape=(kx.size+ky.size),dtype=np.float32)

		descriptor = np.append(kxBoarder,kyBoarder)
		descProject=np.empty(amountNumber,dtype=np.float32);

		for a in range(0,len(A)):

			e=np.dot(A[a],descriptor)
			v=np.power(descriptor,2)
			k=np.sum(descriptor)
			proj=e/k
			descProject[a]=proj

		des1=np.vstack([des1,descProject])

		if(x==(len(kp1)-1)):
			break
			
	return des1
