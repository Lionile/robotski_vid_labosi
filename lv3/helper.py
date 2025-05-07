import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import cv2 as cv

def plot_3d_points(points_3d, plot_show=True):
    """Plot 3D points with different colors based on depth."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use the z-coordinate (depth) for coloring
    z = points_3d[:, 2]
    
    # Scatter plot with depth-based coloring
    scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                         c=z, cmap='viridis', s=20)
    
    # Add labels and color bar
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.colorbar(scatter, ax=ax, label='Depth')
    
    # Set equal aspect ratio
    max_range = np.array([
        points_3d[:, 0].max() - points_3d[:, 0].min(),
        points_3d[:, 1].max() - points_3d[:, 1].min(),
        points_3d[:, 2].max() - points_3d[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
    mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
    mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set view angle
    ax.view_init(elev=30, azim=45)
    
    if plot_show:
        plt.show()
    
    return fig, ax

def remove_outliers(points_3d, threshold=2.0):
    """
    Remove outliers from 3D point cloud using statistical filtering.
    
    Args:
        points_3d: Nx3 array of 3D points
        threshold: Points farther than threshold*std from median are removed
        
    Returns:
        Filtered point cloud without outliers
    """
    # Calculate median point
    median_point = np.median(points_3d, axis=0)
    
    # Calculate distances from median point
    distances = np.sqrt(np.sum((points_3d - median_point) ** 2, axis=1))
    
    # Calculate statistics
    dist_median = np.median(distances)
    dist_std = np.std(distances)
    
    # Filter points within threshold
    mask = distances < (dist_median + threshold * dist_std)
    filtered_points = points_3d[mask]
    
    print(f"Removed {points_3d.shape[0] - filtered_points.shape[0]} outliers out of {points_3d.shape[0]} points")
    
    return filtered_points


def project_points_to_image(points_3d, camera_matrix, img):
    """
    Project 3D points back onto image plane and visualize them colored by depth.
    
    Args:
        points_3d: Nx3 array of 3D points
        camera_matrix: 3x4 camera projection matrix
        img: Original image
    
    Returns:
        Image with projected points colored by depth
    """
    # Convert 3D points to homogeneous coordinates
    points_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    
    # Project points to image plane
    projected_points = camera_matrix @ points_homogeneous.T
    
    # Convert from homogeneous coordinates
    projected_points = projected_points / projected_points[2, :]
    projected_points = projected_points[:2, :].T
    
    # Create a copy of the image to draw on
    result_img = img.copy()
    
    # Get depth values (Z coordinates) and normalize for coloring
    depths = points_3d[:, 2]
    min_depth, max_depth = depths.min(), depths.max()
    normalized_depths = (depths - min_depth) / (max_depth - min_depth)
    
    # Draw projected points with color based on depth
    for i, (x, y) in enumerate(projected_points.astype(int)):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:  # Check if point is within image boundaries
            # Create color from depth using viridis colormap
            depth_normalized = normalized_depths[i]
            color = plt.cm.viridis(depth_normalized)[:3]  # Get RGB from colormap
            color = (int(color[2]*255), int(color[1]*255), int(color[0]*255))  # Convert to BGR for OpenCV
            
            cv.circle(result_img, (x, y), 2, color, -1)
    
    return result_img, (min_depth, max_depth)