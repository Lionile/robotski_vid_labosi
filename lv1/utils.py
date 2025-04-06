import numpy as np
import cv2 as cv
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def shrink_polygon(points, shrink_factor=0.05):
    # Calculate the centroid of the polygon
    centroid_x = sum(p[0] for p in points) / len(points)
    centroid_y = sum(p[1] for p in points) / len(points)
    centroid = (centroid_x, centroid_y)
    
    # Create a new polygon by moving each point toward the centroid
    smaller_points = []
    for point in points:
        vector_x = point[0] - centroid[0]
        vector_y = point[1] - centroid[1]
        
        smaller_x = centroid[0] + vector_x * (1 - shrink_factor)
        smaller_y = centroid[1] + vector_y * (1 - shrink_factor)
        
        smaller_points.append((int(smaller_x), int(smaller_y)))
    
    return smaller_points



def draw_hough_lines(image, lines, color=(0, 0, 255), thickness=2):
    result = image.copy()
    
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            p1 = (int(x0 + 10000 * (-b)), int(y0 + 10000 * (a)))
            p2 = (int(x0 - 10000 * (-b)), int(y0 - 10000 * (a)))
            cv.line(result, p1, p2, color, thickness)
    return result




def estimate_line_strength(edges, rho, theta):
    # Create line mask
    height, width = edges.shape
    line_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Draw the line on the mask with some thickness
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(line_mask, (x1, y1), (x2, y2), 255, 2)
    
    # Count overlapping pixels
    return cv.countNonZero(cv.bitwise_and(edges, line_mask))

def dbscan_edges(values, plot=False):
    cluster_count = 4
    eps = 1.0  # Initial distance threshold
    min_samples = 1  # Minimum samples per cluster
    cluster_centers = []
    original_values = values.copy()
    values = values.squeeze()

    # minmax scaling (values has two columns in each row)
    for column in range(values.shape[1]):
        min_val = np.min(values[:, column])
        max_val = np.max(values[:, column])
        values[:, column] = (values[:, column] - min_val) / (max_val - min_val)
    

    while True:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(values.squeeze())
        unique_clusters = np.unique(clusters[clusters != -1])  # Exclude noise (-1)

        if len(unique_clusters) == cluster_count:
            for i in unique_clusters:
                cluster_lines = original_values[clusters == i].squeeze()
                x_center = np.mean(cluster_lines[:, 0]) if len(cluster_lines.shape) > 1 else cluster_lines[0]
                y_center = np.mean(cluster_lines[:, 1]) if len(cluster_lines.shape) > 1 else cluster_lines[1]
                cluster_centers.append([x_center, y_center])
            break
        elif len(unique_clusters) > cluster_count:
            eps *= 1.1  # Decrease distance threshold
        else:
            eps *= 0.9  # Increase distance threshold

    if plot:
        # Visualize the result on a 2D plane
        plt.figure(figsize=(8, 6))
        for i in unique_clusters:
            cluster_points = original_values[clusters == i].squeeze()
            if len(cluster_points.shape) == 1:  # Handle case with only one cluster point
                cluster_points = cluster_points[np.newaxis, :]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')
        cluster_centers = np.array(cluster_centers)
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', marker='x', s=100, label='Cluster Centers')
        plt.title('DBSCAN Clustering of Hough Lines')
        plt.xlabel('Rho')
        plt.ylabel('Theta')
        plt.legend()
        plt.grid()
        plt.show()

    return cluster_centers