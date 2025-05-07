import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import json
from mpl_toolkits.mplot3d import Axes3D
from helper import plot_3d_points, remove_outliers, project_points_to_image



# Load images
img1 = cv.imread('images/imageL.bmp')
img2 = cv.imread('images/imageR.bmp')
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# Load camera parameters
with open('camera_params_LV5.json', 'r') as f:
    P = np.array(json.load(f)['camera_params'])

# Find features on both images
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# Match features
bf = cv.BFMatcher(cv.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

# Filter only good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Convert keypoints to numpy arrays for findFundamentalMat function
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Find fundamental matrix
F, mask = cv.findFundamentalMat(src_pts, dst_pts, cv.RANSAC, 1.0, 0.99)
print(f"Fundamental matrix:\n{F}")

# Keep only inliers
inlier_mask = mask.ravel().tolist()
good_matches = [good_matches[i] for i in range(len(good_matches)) if inlier_mask[i]]
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

# Calculate essential matrix from fundamental matrix
E = P.T @ F @ P
print(f"Essential matrix:\n{E}")

# Recover pose (rotation and translation) from essential matrix
_, R, t, _ = cv.recoverPose(E, src_pts, dst_pts, P)
print(f"Rotation matrix:\n{R}")
print(f"Translation vector:\n{t}")

# Create projection matrices for both cameras
P1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))  # First camera is at origin [I|0]
P2 = np.hstack((R, t))  # Second camera [R|t]

# Convert to camera matrices by multiplying with intrinsic matrix
M1 = P @ P1
M2 = P @ P2

# Triangulate points
points_4D = cv.triangulatePoints(M1, M2, src_pts.T, dst_pts.T)

# Convert from homogeneous coordinates to 3D coordinates
points_3D = points_4D[:3, :] / points_4D[3, :]
points_3D = points_3D.T

print(f"Original 3D points shape: {points_3D.shape}")

# Remove outliers
filtered_points_3D = remove_outliers(points_3D, threshold=2.0)
print(f"Filtered 3D points shape: {filtered_points_3D.shape}")

# Save both original and filtered 3D points
json.dump(points_3D.tolist(), open('improved_points_3d.json', 'w'))
json.dump(filtered_points_3D.tolist(), open('filtered_points_3d.json', 'w'))

# Visualize matched keypoints
img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                             matchColor=(0, 255, 0), singlePointColor=None, 
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(16, 8))
plt.imshow(cv.cvtColor(img_matches, cv.COLOR_BGR2RGB))
plt.title(f'SIFT Matches: {len(good_matches)} inliers')
plt.tight_layout()

# Plot both original and filtered 3D point clouds
plt.figure(figsize=(18, 8))

# Plot original 3D points
plt.subplot(121, projection='3d')
ax1 = plt.gca()
ax1.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], c=points_3D[:, 2], cmap='viridis', s=20)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title(f'Original 3D Reconstruction ({points_3D.shape[0]} points)')
ax1.view_init(elev=30, azim=45)

# Plot filtered 3D points
plt.subplot(122, projection='3d')
ax2 = plt.gca()
scatter = ax2.scatter(filtered_points_3D[:, 0], filtered_points_3D[:, 1], filtered_points_3D[:, 2], 
                      c=filtered_points_3D[:, 2], cmap='viridis', s=20)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title(f'Filtered 3D Reconstruction ({filtered_points_3D.shape[0]} points)')
ax2.view_init(elev=30, azim=45)

plt.colorbar(scatter, ax=ax2, label='Depth')
plt.tight_layout()


# Project filtered points to both camera views
img1_with_points, depth_range1 = project_points_to_image(filtered_points_3D, M1, img1)
img2_with_points, depth_range2 = project_points_to_image(filtered_points_3D, M2, img2)

# Display results
plt.figure(figsize=(16, 8))

plt.subplot(121)
plt.imshow(cv.cvtColor(img1_with_points, cv.COLOR_BGR2RGB))
plt.title('Left Image with Projected 3D Points')

plt.subplot(122)
plt.imshow(cv.cvtColor(img2_with_points, cv.COLOR_BGR2RGB))
plt.title('Right Image with Projected 3D Points')

# Add a colorbar for depth reference
plt.subplots_adjust(bottom=0.2)
cax = plt.axes([0.15, 0.1, 0.7, 0.05])
cmap = plt.cm.viridis
min_depth = min(depth_range1[0], depth_range2[0])
max_depth = max(depth_range1[1], depth_range2[1])
norm = plt.Normalize(vmin=min_depth, vmax=max_depth)
plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
             cax=cax, orientation='horizontal', 
             label='Depth')

plt.tight_layout()
plt.show()


plt.show()