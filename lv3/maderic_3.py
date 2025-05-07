import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import json
from convert_2d_points_to_3d_points import convert_2d_points_to_3d_points
from plot_3d_points import plot_3d_points
from mpl_toolkits.mplot3d import Axes3D
from helper import remove_outliers



img1 = cv.imread('images/imageL.bmp')
img2 = cv.imread('images/imageR.bmp')
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

with open('camera_params_LV5.json', 'r') as f:
    P = np.array(json.load(f)['camera_params'])


# find features on object
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# match features
bf = cv.BFMatcher(cv.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2) # match all descriptors
# take only best matches
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

# find fundamental matrix
F, mask = cv.findFundamentalMat(src_pts, dst_pts, cv.RANSAC, ransacReprojThreshold=1.0, confidence=0.99)
# print(f'\nFundamental matrix:\n{F}')
matchesMask = mask.ravel().tolist()
h, w = img1_gray.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
# find transform with given fundamental matrix
dst = cv.perspectiveTransform(pts,F)

# plot good sift matches
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
img = cv.drawMatches(img1_gray, kp1, img2_gray, kp2, good, None, **draw_params)
plt.imshow(img)


# essential matrix
E = P.T @ F @ P
# print(f'\nEssential matrix:\n{E}')
src_pts = np.array([ kp1[m.queryIdx] for m in good ])
dst_pts = np.array([ kp2[m.trainIdx] for m in good ])
# print(f'src: \n{src_pts[0].pt}\n\ndst: \n{dst_pts[:10]}')
points_3d = convert_2d_points_to_3d_points(src_pts, dst_pts, E, P)

json.dump(points_3d.tolist(), open('points_3d.json', 'w'))

plot_3d_points(points_3d, plot_show=False)
filtered_points_3d = remove_outliers(points_3d, threshold=.3)
plot_3d_points(filtered_points_3d, plot_show=False)


plt.show()
