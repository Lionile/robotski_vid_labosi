import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random
import time
import os
from scipy.optimize import minimize
from multiprocessing import Pool
from read_kinect_pic import read_kinect_pic


def point_cloud_ransac(depth_image, eps, iterations=100):
    T = [] # list of inlier points
    R = [0, 0, 0] # plane paramaters (au + bv + c = d)
    height, width = depth_image.shape

    for _ in range(iterations):
        # pick three distinct points from the point cloud
        indices = random.sample(range(width * height), 3)
        
        p1 = (indices[0]//width, indices[0]%height)
        p2 = (indices[1]//width, indices[1]%height)
        p3 = (indices[2]//width, indices[2]%height)
        
        d1 = depth_image[*p1]
        d2 = depth_image[*p2]
        d3 = depth_image[*p3]

        # calculate plane parameters from the three points
        # A * R = d
        A = np.array([[*p1, 1],
                    [*p2, 1],
                    [*p3, 1]])
        d = np.array([d1, d2, d3])

        try: # try catch for matrix singularity
            R_ = np.linalg.inv(A) @ d
        except np.linalg.LinAlgError:
            continue
        inlier_points = np.array([])
        for u, row in enumerate(depth_image):
            for v, pixel in enumerate(row):
                # distance of point to the plane
                dist = np.abs(pixel - (R_[0]*u + R_[1]*v + R_[2]))
                if dist < eps:
                    inlier_points = np.vstack([inlier_points, [u, v]]) if len(inlier_points) > 0 else np.array([u, v])
        # print(f'inlier count: {len(inlier_points)}')
        if len(inlier_points) > len(T):
            T = inlier_points
            R = R_
            # print(f'best inlier count: {len(T)}')
    
    return T, R

def ransac_worker(args):
    T, R = point_cloud_ransac(*args)
    return T, R

if __name__ == '__main__':
    # image_numbers = [1, 2, 3, 133, 242, 270, 300, 392, 411]
    image_numbers = [133]
    num_planes = 1

    images_folder_path = 'result_images'
    if not os.path.exists(images_folder_path):
        os.mkdir(images_folder_path)

    for image_num in image_numbers:
        img_filename = rf'sl-{image_num:05d}.bmp'
        depth_img_filename = rf'sl-{image_num:05d}-D.txt'
        img = cv.imread(rf'images\{img_filename}')
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        depth_img, point_3d_array, n_3d_points = read_kinect_pic(rf'images\{depth_img_filename}', img.shape[:2])
        # single threaded
        # T, R = point_cloud_ransac(depth_image, 3, 500)


        # multi threaded
        for eps in [3]:
            start_time = time.time()
            # algorithm params
            iterations = 40 # per process
            # multithreading params
            num_workers = 8
            with Pool(num_workers) as pool:
                results = pool.map(ransac_worker, [(depth_img, eps, iterations)] * num_workers)

            T, R = max(results, key=lambda x: len(x[0]))
            print(f'Inlier count: {len(T)}\nBest R: {R}')
            end_time = time.time()
            print(f'Execution time: {(end_time - start_time):.2f}s')
            

            plane_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

            img_dominant = img.copy()
            for px in T:
                img_dominant[*px] = (255, 0, 0)

            depth_img_dominant = cv.cvtColor(depth_img, cv.COLOR_GRAY2RGB)
            for px in T:
                depth_img_dominant[*px] = (255, 0, 0)

            fig, ax = plt.subplots(2, 2, figsize=(12, 6))
            ax[0,0].imshow(img)
            ax[0,0].set_title('original image')
            ax[0,1].imshow(img_dominant)
            ax[0,1].set_title('original image with dominant plane')

            ax[1,0].imshow(depth_img, cmap='gray')
            ax[1,0].set_title('original depth image')
            ax[1,1].imshow(depth_img_dominant)
            ax[1,1].set_title('depth image with dominant plane')
            plt.suptitle(f'eps: {eps}')
        
            # fine tune with local optimization method
            # def calculate_inliers(depth_img, R, eps):
            #     inlier_points = np.array([])
            #     for u, row in enumerate(depth_img):
            #         for v, pixel in enumerate(row):
            #             # distance of point to the plane
            #             dist = np.abs(pixel - (R[0]*u + R[1]*v + R[2]))
            #             if dist < eps:
            #                 inlier_points = np.vstack([inlier_points, [u, v]]) if len(inlier_points) > 0 else np.array([u, v])
            #     return inlier_points
            # def objective(R):
            #     global depth_img
            #     return -len(calculate_inliers(depth_img, R, eps))
            
            # R_fine = minimize(objective, x0=R, method='Nelder-Mead').x
            # print(f'Inlier count: {len(T)}\nfine tuned R: {R_fine}')

            # inlier_points = calculate_inliers(depth_img, R_fine, eps)
            # depth_image_dominant_fine = cv.cvtColor(depth_img, cv.COLOR_GRAY2RGB)
            # for px in T:
            #     depth_image_dominant_fine[*px] = (255, 0, 0)
            
            # ax[,].imshow(depth_image_dominant_fine, cmap='gray')
            # ax[,].set_title('depth image with fine tuned plane parameters')

            # save resulting image
            # cv.imwrite(rf'{images_folder_path}/{img_filename}_eps{eps}.png', cv.cvtColor(img_dominant, cv.COLOR_BGR2RGB)) # original with dominant plane
            # cv.imwrite(rf'{images_folder_path}/{img_filename}_eps{eps}_depth.png', cv.cvtColor(depth_img_dominant, cv.COLOR_BGR2RGB)) # depth with dominant plane
            # report_img = np.hstack([img, img_dominant, depth_img_dominant])
            # cv.imwrite(rf'{images_folder_path}/{img_filename}_eps{eps}_report.png', cv.cvtColor(report_img, cv.COLOR_BGR2RGB)) # depth with dominant plane
        
    plt.tight_layout()
    plt.show()