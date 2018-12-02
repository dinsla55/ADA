
import numpy as np

def getProjectEig(numberProject):
	text_file = open("gpcavects.txt","r")
	lines = text_file.read().split(' ')
	linesNew= list(filter(None,lines))

	project= np.empty(shape=(numberProject,3042),dtype=np.float32)
	for i in range(0,numberProject-1):
		for j in range(0,3041):
			project[i][j]=linesNew[i+(j*i)]


	text_file.close()

	return project
