import numpy as np
import cv2
from matplotlib import pyplot as plt
def findAffineTransform(img1, img2):

    # define constants
    MIN_MATCH_COUNT = 10
    MIN_DIST_THRESHOLD = 0.7
    RANSAC_REPROJ_THRESHOLD = 5.0

    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m in matches:
    # 100 just a value should be maximal n.distance
        if m.distance <100*MIN_DIST_THRESHOLD:
            good.append(m)
    res=cv2.drawMatches(img1,kp1,img2,kp2,good,None,flags=2)
    cv2.imshow("MATCHES",res)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M,tmp = cv2.findHomography(src_pts, dst_pts)
        return M;

    else: raise Exception("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
