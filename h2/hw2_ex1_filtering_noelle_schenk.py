""" 1D/2D convolution and Filtering """

# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
plt.rcParams['image.cmap'] = 'gray'
import time
import pdb
from tools_template import *
import skimage.io as io

# import os
# os.getcwd()
# os.chdir('/home/exserta/Documents/signal_and_image_processing/h2/hw2_material')

############################ Implement 1D Convolution ###########################
# 1.1. Implement 1D convolution
# Implement your myconv() function in tools_template.py by filling in the template
# And use your function myconv to filter a signal and plot it
sig = np.repeat([0., 1., 0.], 100)
win = signal.hann(50)
filtered = myconv(sig, win)
plt.figure()
ax1 = plt.subplot(311)
plt.subplot(311)
plt.plot(sig)
plt.title('Original signal')
plt.subplot(312, sharex=ax1)
plt.plot(win)
plt.title('Filter')
plt.subplot(313, sharex=ax1)
plt.plot(filtered)
plt.title('Filtered signal')
plt.show()

############################## Implement 2D Convolution ######################

# 1.2 Implement 2D Convolution
# Implement your myconv2() function in tools_template.py
# have a laplacian filter to test around
lap2 = np.array([[1, 4, 1], [4, -20, 4], [1, 4, 1]])

# 1.3 Implement Gaussian Filtering
# Implement your gconv() function in tools_template.py

# 1.4. Blur an image with a Gaussian filter
# Use your code gconv on the image for some sigma value and display the result
img = plt.imread('ex1_img.jpg').astype(np.float32)
# plt.imshow(img)
sigma = 3
filter_size = 9
img_filtered = gconv(img, sigma, filter_size)
img_filtered2 = gconv(img, 9, 18)

plt.subplot(1,3,1)
plt.imshow(img)
plt.axis('off')
plt.title('original image')
plt.subplot(1, 3, 2)
plt.imshow(img_filtered)
plt.axis('off')
plt.title('Image filtered with \n gaussian filter of sigma = {}'.format(str(sigma)))
plt.subplot(1, 3, 3)
plt.imshow(img_filtered2)
plt.axis('off')
plt.title('Image filtered with \n gaussian filter of sigma 9, size 18')
plt.show()
