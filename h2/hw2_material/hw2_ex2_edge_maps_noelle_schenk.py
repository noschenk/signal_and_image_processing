""" Use of Difference of Gaussian to create edge maps """

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tools_template import *

# Read the original image
img = Image.open('ex2_img.png').convert('L')
img = np.asarray(img)
# convert to float32
img = np.float32(img / 256)

# 2.1. Implement DoG
# Implement function DoG in tools_template.py

# 2.2. Create DoG of an image with given sigma and filter size values
sigma_1 = 1
sigma_2 = 1.5
filter_size = 30
img_filtered_1 = gconv(img, sigma_1, filter_size)
img_filtered_2 = gconv(img, sigma_2, filter_size)
# calculate DoG of the image with my function
DoG_of_img = DoG(img, sigma_1, sigma_2, filter_size)

plt.figure(figsize=(7, 7))
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original image')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(img_filtered_1)
plt.axis('off')
plt.title('Filtered image with sigma: {:.1f}'.format(sigma_1))
plt.subplot(2, 2, 3)
plt.imshow(img_filtered_2)
plt.axis('off')
plt.title('Filtered image with sigma: {:.1f}'.format(sigma_2))
plt.subplot(2, 2, 4)
plt.imshow(DoG_of_img)
plt.title('Edge map by DoG')
plt.axis('off')
plt.savefig('edge_map_img.png')
plt.show()

# 2.3. Do the same thing with different parameters
sigma_1 = 1
sigma_2 = 25
filter_size = 30
img_filtered_1 = gconv(img, sigma_1, filter_size)
img_filtered_2 = gconv(img, sigma_2, filter_size)
DoG_of_img = DoG(img, sigma_1, sigma_2, filter_size)

plt.figure(figsize=(7, 7))
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original image')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(img_filtered_1)
plt.axis('off')
plt.title('Filtered image with sigma: {:.1f}'.format(sigma_1))
plt.subplot(2, 2, 3)
plt.imshow(img_filtered_2)
plt.axis('off')
plt.title('Filtered image with sigma: {:.1f}'.format(sigma_2))
plt.subplot(2, 2, 4)
plt.imshow(DoG_of_img)
plt.title('Edge map by DoG')
plt.axis('off')
plt.savefig('edge_map_img.png')
plt.show()
