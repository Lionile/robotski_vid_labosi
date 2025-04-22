import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


image_res = 12 # 12 or 50 Mpx
original_image = f'images/image1_{image_res}.jpg'
images_to_search = [f'images/image2_{image_res}.jpg', f'images/image3_{image_res}.jpg', f'images/image4_{image_res}.jpg', f'images/image5_{image_res}.jpg']

img1 = cv.imread(original_image)
# select region (object) of interest on image
# Resize image for ROI selection
display_scale = 0.25  # Adjust this value as needed
img1_small = cv.resize(img1, (0, 0), fx=display_scale, fy=display_scale)

# select region (object) of interest on resized image
roi_small = cv.selectROI(img1_small)
# Scale ROI coordinates back to original image size
roi = (
    int(roi_small[0] / display_scale),
    int(roi_small[1] / display_scale),
    int(roi_small[2] / display_scale),
    int(roi_small[3] / display_scale)
)
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img1_roi = img1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
img1_roi_gray = img1_gray[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

# clahe
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# img1_roi_gray = clahe.apply(img1_roi_gray)



# find features on object
sift = cv.SIFT_create(nfeatures=10000)
kp1, des1 = sift.detectAndCompute(img1_roi_gray, None)

# for each image find the selected object
final_images = []
for img_file in images_to_search:
    img2 = cv.imread(img_file)
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    # img2_gray = clahe.apply(img2_gray)

    # show features
    # img = cv.drawKeypoints(img1_roi, kp, img1_roi)
    # cv.imshow('image', img)
    # cv.waitKey(0)

    # find all features on second image
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # match features
    bf = cv.BFMatcher(cv.NORM_L2)

    matches = bf.knnMatch(des1, des2, k=2) # match all descriptors
    # take only best matches
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    # find homography and draw bounding box
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h, w = img1_roi_gray.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img_bbox = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
    img = cv.drawMatches(img1_roi, kp1, img_bbox, kp2, good, None, **draw_params)
    final_images.append(img)


_, ax = plt.subplots(len(images_to_search)//2, 2)

for i in range(len(images_to_search)):
    ax[i%2, i//2].imshow(cv.cvtColor(final_images[i], cv.COLOR_BGR2RGB), 'gray')

plt.show()