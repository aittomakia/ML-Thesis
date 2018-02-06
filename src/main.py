# Python 2 and 3 compatibility
from __future__ import print_function
import os
# os.path.dirname(os.path.abspath(__file__))
import sys
PY3 = sys.version_info[0] == 3
sys.path.append('/usr/local/lib/python3.6/site-packages')

if PY3:
    from functools import reduce

import os.path
import cv2
import numpy as np
from matplotlib import pyplot as plt

SZ = 20
CLASS_N = 10

def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)
    print(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    print("M", M)
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    # img = cv2.warpAffine(img, M, (75, 48), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def load_images(fn):
    digits_img = cv2.imread(fn, 0)
    # digits = split2d(digits_img, (SZ, SZ))
    # digits = split2d(digits_img, (48, 75))
    # labels = np.repeat(np.arange(CLASS_N), len(digits)/CLASS_N)
    # return digits, labels
    return digits_img

# HOG = Histogram of Oriented Gradients
def get_hog() :
    winSize = (20,20) # Same size as the SZ
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

    return hog
    affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR

#while not os.path.isfile('button.png'):
    #ignore if no such file is present.
#    pass

# Load data.
# images, labels = load_images('button2.png')
# subprocess.call(['firefox', images])

# Deskew image
# image_deskewed = deskew, images
# print(image_deskewed)

# img, labels = load_images('button2.png')
img = load_images('button.png')
print(img)
# edges = cv2.Canny(img, 1024, 768, L2gradient=False)

plt.subplot(131), plt.imshow(img, cmap = 'gray')
plt.title('Alkuperainen kuva'), plt.xticks([]), plt.yticks([])

# plt.subplot(132), plt.imshow(edges, cmap = 'gray')
# plt.title('Reunat tunnistettu'), plt.xticks([]), plt.yticks([])

# img_deskewed = deskew(edges)
# img_deskewed = deskew(img)

# cv2.imwrite("unskewed.png", img_deskewed)
# cv2.imshow("Vis", img_deskewed)

# plt.subplot(133), plt.imshow(img_deskewed, cmap = 'gray')
# plt.title('Ei-vaaristetty'), plt.xticks([]), plt.yticks([])
#plt.subplot(122), plt.imshow(edges)
#plt.title('Reunat tunnistettu'), plt.xticks([]), plt.yticks([])
plt.show()


# 1. Import library of functions
import tflearn

# 2. Logical OR operator / the data
OR = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
Y_truth = [[0.], [1.], [1.], [1.]]

# 3. Building our neural network/layers of functions
neural_net = tflearn.input_data(shape=[None, 2])
neural_net = tflearn.fully_connected(neural_net, 1, activation='sigmoid')
neural_net = tflearn.regression(neural_net, optimizer='sgd', learning_rate=2, loss='mean_square')

# 4. Train the neural network / Epochs
m = tflearn.DNN(neural_net)
m.fit(OR, Y_truth, n_epoch=2000, snapshot_epoch=False)

# 5. Testing final prediction
print("Testing OR operator")
print("0 or 0:", m.predict([[0., 0.]]))
print("0 or 1:", m.predict([[0., 1.]]))
print("1 or 0:", m.predict([[1., 0.]]))
print("1 or 1:", m.predict([[1., 1.]]))
