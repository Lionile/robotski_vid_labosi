import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('./images/image5.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# blurred = cv.GaussianBlur(gray, (5,5), 10)
# for _ in range(100):
#     blurred = cv.GaussianBlur(blurred, (5,5), 10)


fig, ax = plt.subplots(2, 4, figsize=(12, 8))
for i in range(8):
    range = i
    canny = cv.Canny(gray, range*20, range*20+20)

    ax[i//4, i%4].imshow(canny, cmap='gray')

plt.show()
