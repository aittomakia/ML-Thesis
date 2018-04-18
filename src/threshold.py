import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('src/static/upload/Thesis-Test.jpg',0)
img = cv.medianBlur(img,5)
ret,th1 = cv.threshold(img,130,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)']
images = [img, th1]

height, width = th1.shape[:2]
max_height = 600
max_width = 600

# only shrink if the image is bigger than the max values
if max_height < height or max_width < width:
    # get scaling factor
    scaling_factor = max_height / float(height)
    if max_width/float(width) < scaling_factor:
        scaling_factor = max_width / float(width)
    # resize image
    th1 = cv.resize(th1, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_AREA)

cv.imshow('result', th1)
cv.waitKey(0)
cv.destroyAllWindows()


#for i in range(2):
#    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#    plt.title(titles[i])
#    plt.xticks([]),plt.yticks([])
#plt.show()
