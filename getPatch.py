
import numpy as np
import tkinter
import cv2
from matplotlib import pyplot as plt
from unpackSift import unpackSIFTOctave
from matplotlib.mlab import PCA
from eigenVec import getProjectEig
import math


def getPatch(keypoint,img,patchsize):
	oct , layer, scale=unpackSIFTOctave(keypoint)
	patchsizeQ=int(round(patchsize*math.sqrt(2)/2*scale))
	if(patchsizeQ==0):
		patchsizeQ=1

	rPatch = np.empty(shape=(patchsizeQ*2,patchsizeQ*2,3),dtype=np.float32)

	rPatch=img[int(keypoint.pt[0]-patchsizeQ):int(keypoint.pt[0]+patchsizeQ) ,
			int(keypoint.pt[1]-patchsizeQ):int(keypoint.pt[1]+patchsizeQ)]

	res = cv2.resize(rPatch,None,fx=1/scale,fy=1/scale,interpolation=cv2.INTER_CUBIC)
	M=cv2.getRotationMatrix2D((int(patchsizeQ/scale),int(patchsizeQ/scale)),keypoint.angle,1)

	dst = cv2.warpAffine(res,M,res.shape)


	patch = np.empty(shape=(patchsize,patchsize,3),dtype=np.float32)
	rowsPa, colsPa=dst.shape
	patch=dst[int(patchsizeQ/scale)-patchsize//2:int(patchsizeQ/scale)+patchsize//2+1 ,
			int(patchsizeQ/scale)-patchsize//2:int(patchsizeQ/scale)+patchsize//2+1]

	return patch
