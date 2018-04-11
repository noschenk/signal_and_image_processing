""" Face detector using template matching """

import numpy as np
from PIL import Image
from skimage.feature import match_template
import matplotlib.pyplot as plt
from tools_template import *
plt.rcParams['image.cmap'] = 'gray'
import scipy.misc as misc

# Read the image and the template
img = np.asarray(Image.open('ex3_img.jpg').convert('L'))
template = np.asarray(Image.open('ex3_template.jpg').convert('L'))


# 3.1 Implement Gaussian Pyramid
# 3.1.a Write your function blur_and_donwsample() in tools_template.py
# 3.1.b Write your function generate_gaussian_pyramid() in tools_template.py
# 3.1.c Create Guasian Pyramid and show the result
sigma = 4
scale = 0.6
filter_size = 8
num_levels = 4
gaussian_pyramid = generate_gaussian_pyramid(img, sigma, filter_size, scale, num_levels)


n_plots = np.ceil(np.sqrt(num_levels))
for k, gp in enumerate(gaussian_pyramid):
    plt.subplot(n_plots, n_plots, k+1)
    plt.imshow(gp)
    plt.axis('off')
    if k==0:
        plt.title('Original image - Level-{:d}'.format(k))
    else: 
        plt.title('Level-{:d}'.format(k))
plt.show()


# 3.2 Create Gaussian Pyramid on the template
sigma = 3
scale = 0.8
filter_size = 8
num_levels = 3
gaussian_pyramid = generate_gaussian_pyramid(template, sigma, filter_size, scale, num_levels)
plt.imshow(gaussian_pyramid[1])

# 3.3 Run the built-in match_template function on the gaussian pyramid images and
# the template
# creating the list corr_mat_list which contains correlation matrices for every
#    template level with the original image.
corr_mat_list = list()
for i in range(0, num_levels):
    m = match_template(img, gaussian_pyramid[i], pad_input=True)
    corr_mat_list.append(m)

plt.figure(figsize = (7, 7))
n_plots = np.ceil(np.sqrt(num_levels+1))
plt.subplot(n_plots, n_plots, 1)
plt.imshow(img)
for k, corr_mat in enumerate(corr_mat_list):
    plt.subplot(n_plots, n_plots, k+2)
    plt.imshow(corr_mat)
    plt.title('Normalized Corr. Matrix \n Level {:d}'.format(k))
plt.show()


# 3.4 Threshold the resulting correlation matrices
threshold = 0.56
# create the list binary_images in which every image is converted into a binary with
#    values 0 if the NCC is below the treshold or value 1 if the NCC is above treshold.
binary_images = list()
for i in range(0, num_levels):
    b = corr_mat_list[i]
    b[b < threshold] = 0
    b[b >= threshold] = 1
    binary_images.append(b)

plt.figure(figsize = (7, 7))
n_plots = np.ceil(np.sqrt(num_levels+1))
plt.subplot(n_plots, n_plots, 1)
plt.imshow(img)
plt.axis('off')
plt.title('Original image')
for k, bn_img in enumerate(binary_images):
    plt.subplot(n_plots, n_plots, k+2)
    plt.imshow(bn_img)
    plt.axis('off')
    plt.title('Binary image Level {:d}'.format(k))
plt.show()

plt.figure(figsize = (9, 7))
n_plots = np.ceil(np.sqrt(num_levels+1))
plt.subplot(n_plots, n_plots, 1)
plt.imshow(img)
plt.title('Original image')
plt.axis('off')
for k, bn_img in enumerate(binary_images):
    plt.subplot(n_plots, n_plots, k+2)
    plt.imshow(img)
    plt.imshow(bn_img, alpha = 0.5)
    plt.axis('off')
    plt.title('Binary image on top of original image \n Level {:d}'.format(k))
plt.show()

# 3.5 Draw rectangles around the detected faces
plt.figure(figsize = (7, 7))
n_plots = np.ceil(np.sqrt(num_levels+1))
plt.subplot(n_plots, n_plots, 1)
plt.imshow(img)
plt.axis('off')
plt.title('Original image')
for k, bn_img in enumerate(binary_images):
    ax = plt.subplot(n_plots, n_plots, k+2)
    ax.imshow(img)
    idx = np.where(bn_img)
    for n in range(len(idx[0])):
        h, w = gaussian_pyramid[k].shape
        rect = plt.Rectangle((idx[1][n]-w/2, idx[0][n]-h/2), w, h, edgecolor = 'r', facecolor='none')
        ax.add_patch(rect)
    plt.title('Detected faces \n Level {:d}'.format(k))
    plt.axis('off')
plt.show()

