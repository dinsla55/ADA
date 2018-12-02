
import numpy as np

def getPercent(M,src_pts,dst_pts):

    counter=0;
    for i in range(0,len(src_pts)):
        hx=(M[0][0]*src_pts[i][0]+M[0][1]*src_pts[i][1]+M[0][2])
        hy=(M[1][0]*src_pts[i][0]+M[1][1]*src_pts[i][1]+M[1][2])
        hz=(M[2][0]*src_pts[i][0]+M[2][1]*src_pts[i][1]+M[2][2])
        xnew=hx/hz
        ynew=hy/hz
        pt_new=np.array([xnew ,ynew])
        dif=np.subtract(dst_pts[i],pt_new)

        pow=np.power(dif,2)

        summe=np.sum(pow)

        if(summe<10):
            counter+=1
    return (counter/len(src_pts))
