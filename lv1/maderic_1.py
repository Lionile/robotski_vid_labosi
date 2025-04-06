import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import json
from utils import draw_hough_lines, shrink_polygon, estimate_line_strength, dbscan_edges

cap = cv.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")


# take image of paper
while True:
    c = cv.waitKey(15)

    ret, frame = cap.read()
    cv.imshow('Input', frame)
    image_size = frame.size

    if c == ord('c'):
        cv.imwrite('./images/image.jpg', frame)

    if c == 27: # escape
        break
cv.destroyAllWindows()


# user selects paper corners
points = []
max_points = 4
image = None

# Mouse callback function to handle point selection
def select_point(event, x, y, flags, param):
    global points, image
    
    if event == cv.EVENT_LBUTTONDOWN:
        if len(points) < max_points:
            points.append((x, y))
            # Draw circle at the selected point
            cv.circle(image, (x, y), 2, (0, 0, 255), -1)
            cv.imshow('Select Points', image)
            print(f"Point {len(points)} selected: ({x}, {y})")
            
            if len(points) == max_points:
                print("All 4 points selected")

original_image = cv.imread('./images/image.jpg')
image = original_image.copy()
with open('camera_params.json', 'r') as f:
        data = json.load(f)

# undistort
image = cv.undistort(image, np.array(data["camera_matrix"]), np.array(data["dist_coeffs"]), None)

# Create window and set mouse callback
cv.namedWindow('Select Points', cv.WINDOW_NORMAL)
cv.resizeWindow('Select Points', 640, 480)
cv.setMouseCallback('Select Points', select_point)
cv.imshow('Select Points', image)

# wait until max_points selected on the image
while len(points) < max_points:
    key = cv.waitKey(10)
    if key == 27:  # escape
        break
cv.destroyAllWindows()

# remove everything outside of selected paper points
mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv.fillPoly(mask, np.array([points]), 1)
image = cv.bitwise_and(image, image, mask=mask)



# take one (first) point as the origin on the paper
origin_paper = points[0]

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# blurring
# blurred = cv.GaussianBlur(gray, (5,5), 10)
# for _ in range(25):
#    blurred = cv.GaussianBlur(blurred, (5,5), 10)



edges = cv.Canny(gray, 140, 160) # get edges

# remove outer edges from paper
inner_points = shrink_polygon(points, 0.05)
inner_mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv.fillPoly(inner_mask, np.array([inner_points]), 1)
paper_edges = cv.bitwise_and(edges, edges, mask=1-inner_mask) # inverted mask to ger paper edges
# plt.imshow(paper_edges); plt.show()
edges = cv.bitwise_and(edges, edges, mask=inner_mask)


lines = cv.HoughLines(edges, 1, np.pi/90, threshold=50) # get lines
paper_lines = cv.HoughLines(paper_edges, 1, np.pi/90, threshold=50) # get paper lines


# draw best match hough lines on the image
# find 4 groups with dbscan
cluster_centers = dbscan_edges(lines)
cluster_centers_paper = dbscan_edges(paper_lines)


# sort lines by number of votes
# line_strengths = [estimate_line_strength(paper_edges, line[0], line[1]) for line in cluster_centers_paper]
# sorted_indices = np.argsort(line_strengths)[::-1]
# cluster_centers_paper = np.array(cluster_centers_paper)[sorted_indices].squeeze()

# display detected object lines
fig, ax = plt.subplots(1, 2)
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
hough_image = image.copy()
for i in range(len(cluster_centers)):
    hough_image = draw_hough_lines(hough_image, [[cluster_centers[i]]], color=colours[i])
ax[0].imshow(hough_image)
ax[1].imshow(edges, cmap='gray')
plt.show()

# get middle of the object
# convert lines from polar to slope-intercept
lines_cartesian = []
# sort clusters by angle for grouping lines in the same direction
cluster_centers = np.array(cluster_centers)
argsort = cluster_centers[:,1].argsort()
cluster_centers = cluster_centers[argsort]
for i in range(len(cluster_centers)):
    rho = cluster_centers[i,0]
    theta = cluster_centers[i,1]
    lines_cartesian.append([-np.cos(theta)/np.sin(theta), rho/np.sin(theta)]) # mx + c -> [m, c]


# display object edges and center relative to the paper edge
# object_edges = []
# for i in range(2):
#     for j in range(2):
#         m1, c1 = lines_cartesian[i]
#         m2, c2 = lines_cartesian[j+2]
#         x = (c2 - c1) / (m1 - m2)
#         y = m1 * x + c1
#         object_edges.append([x, y])
# object_edges = np.array(object_edges)

# object_center_image = [int(np.average(object_edges[:,0])), int(np.average(object_edges[:,1]))]

# plt.imshow(image)
# plt.scatter(object_center_image[0], object_center_image[1], marker='*')
# plt.plot([points[0][0], object_center_image[0]], [points[0][1], object_center_image[1]])
# plt.show()



# determine ro and theta in the coordinate system of the paper
P = np.array(data["camera_matrix"])

# align paper with camera
# solvepnp
_, rvec, tvec = cv.solvePnP(np.array([[0.0, 0.0, 0.0], [0.0, 190, 0.0], [277, 190, 0.0], [277, 0.0, 0.0]], dtype=np.float32), np.array(points, dtype=np.float32), P, np.array(data["dist_coeffs"]))
# get rotation matrix from rvec
R, _ = cv.Rodrigues(rvec)
R = np.array(R, dtype=np.float32)

A = P @ R
b = P @ tvec

# get dominant line (closest to the paper origin)
cluster_centers = np.array(cluster_centers)
sorted_indices = np.argsort(cluster_centers[:,0])
cluster_centers = np.array(cluster_centers)[sorted_indices].squeeze()
dominant_line = cluster_centers[-1]

# display dominant line on the image
hough_image = image.copy()
hough_image = draw_hough_lines(hough_image, [[dominant_line]], color=(255, 0, 0))
plt.imshow(hough_image)
plt.show()

rho, theta = np.array(dominant_line).squeeze()

lambda_x = A[0][0] * np.cos(theta)+A[1][0] * np.sin(theta) - rho * A[2][0]
lambda_y = A[0][1] * np.cos(theta)+A[1][1] * np.sin(theta) - rho * A[2][1]
lambda_p = b[2] * rho - b[0] * np.cos(theta) - b[1] * np.sin(theta)

theta_prime = np.arctan2(lambda_y, lambda_x)
rho_prime = lambda_p / (np.sqrt(lambda_x**2 + lambda_y**2))

print(f'rho: {rho_prime}, theta: {np.degrees(theta_prime)}')

real_rho = 20 # mm
real_angle = -58 # degrees

rho_diff = abs(real_rho - rho_prime)
angle_diff = abs(real_angle - np.degrees(theta_prime))
print(f'accuracy:\nrho: {rho_diff} mm ({rho_diff/real_rho * 100}%)\nangle: {angle_diff} degrees ({angle_diff/real_angle * 100}%)')