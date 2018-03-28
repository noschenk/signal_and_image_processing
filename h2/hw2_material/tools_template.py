""" 
Contains functions that are mainly used in all exercises of HW2
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import time
from scipy.misc import imresize
from scipy.signal import convolve2d
import scipy.ndimage as ndi
import scipy.signal as signal

def gauss1d(sigma, filter_length=10):
    # INPUTS
    # @ sigma         : standard deviation of gaussian distribution
    # @ filter_length : integer denoting the filter length, default is 10
    # OUTPUTS
    # @ gauss_filter  : 1D gaussian filter

    # if filter_length is even add one
    filter_length += ~filter_length % 2
    x = np.linspace(np.int(-filter_length/2),np.int(filter_length/2), filter_length)

    gauss_filter = np.exp(- (x ** 2) / (2 * (sigma ** 2)))

    gauss_filter = gauss_filter / np.sum(gauss_filter)
    
    return gauss_filter

def gauss2d(sigma, filter_size=10):
    # INPUTS
    # @ sigma           : standard deviation of gaussian distribution
    # @ filter_size     : integer denoting the filter size, default is 10
    # OUTPUTS
    # @ gauss2d_filter  : 2D gaussian filter

    # create a 1D gaussian filter
    gauss1d_filter = gauss1d(sigma, filter_size)[np.newaxis, :]
    # convolve it with its transpose
    gauss2d_filter = convolve2d(gauss1d_filter, np.transpose(gauss1d_filter))
    
    return gauss2d_filter

def myconv(signal, filt):
    # This function performs a 1D convolution between signal and filt. This
    # function should return the result of a 1D convolution of the signal and
    # the filter.
    # INPUTS
    # @ signal          : 1D image, as numpy array, of length m
    # @ filt            : 1D or 2D filter of length k
    # OUTPUTS
    # signal_filtered   : 1D filtered signal, of size (m+k-1)
    res = np.convolve(signal, filt, mode='full')
    return res

    pass

def myconv2(img, filt):
    # This function performs a 2D convolution between image and filt, image being a 2D image. This
    # function should return the result of a 2D convolution of these two
    # images. NOTE: If you can not implement 2D convolution, you can use
    # scipy.signal.convolve2d() in order to be able to continue with the other exercises.
    # INPUTS
    # @ img           : 2D image, as numpy array, size mxn
    # @ filt          : 1D or 2D filter of size kxl
    # OUTPUTS
    # img_filtered    : 2D filtered image, of size (m+k-1)x(n+l-1)
    if img.dtype == 'float32':
        res = convolve2d(img, filt, mode='full')
        # have the image in the appropriate range again
        res = (res - res.min()) / (res.max() - res.min())
    else:
        print('convert your image into float32')
    return np.float32(res)

    pass

def gconv(img, sigma, filter_size):
    # Function that filters an image with a Gaussian filter
    # INPUTS
    # @ img           : 2D image
    # @ sigma         : the standard deviation of gaussian distribution
    # @ size          : the size of the filter
    # OUTPUTS
    # @ img_filtered  : filtered image with gaussian filter

    # create a gaussian filter gk ( = gaussian kernel) with given sigma :
    arm = ((filter_size - (filter_size % 2)) / 2)
    a = np.linspace(-arm, arm, filter_size)
    c, r = np.meshgrid(a, a)
    gk = np.exp(-(c ** 2 + r ** 2) / (2 * sigma ** 2))
    # apply it on the image
    res = myconv2(img, gk)
    return res

    pass

def DoG(img, sigma_1, sigma_2, filter_size):
    # Function that creates Difference of Gaussians (DoG) for given standard
    # deviations and filter size
    # INPUTS
    # @ img
    # @ img           : 2D image (MxN)
    # @ sigma_1       : standard deviation of of the first Gaussian distribution
    # @ sigma_2       : standard deviation of the second Gaussian distribution
    # @ filter_size   : the size of the filters
    # OUTPUTS
    # @ dog           : Difference of Gaussians of size
    #                   (M+filter_size-1)x(N_filter_size-1)
    # if the filter has an even number, numpy automatically takes one value less.
    if filter_size % 2 == 0:
        filter_size -= 1
        print('filter_size changed to :', filter_size)
    img_filtered_1 = gconv(img, sigma_1, filter_size)
    img_filtered_2 = gconv(img, sigma_2, filter_size)
    # pad the original image with zeros so it has the same shape as the filtered ones
    padwidth = int((filter_size - (filter_size % 2)) / 2)
    img_padded = np.pad(img, pad_width=padwidth, mode='constant')
    # calculate the difference of the filtered images
    resimg = img_filtered_1 - img_filtered_2
    return resimg

    pass

 def blur_and_downsample(img, sigma, filter_size, scale):
    # INPUTS
    # @ img                 : 2D image (MxN)
    # @ sigma               : standard deviation of the Gaussian filter to be used at
    #                         all levels
    # @ filter_size         : the size of the filters
    # @ scale               : Downscaling factor (of type float, between 0-1)
    # OUTPUTS
    # @ img_br_ds           : The blurred and downscaled image 
    
    pass       

def generate_gaussian_pyramid(img, sigma, filter_size, scale, num_levels):
    # Function that creates Gaussian Pyramid as described in the homework
    # It blurs and downsacle the iimage subsequently. Please keep in mind that
    # the first element of the pyramid is the oirignal image, which is
    # considered as the level-0. The number of levels that is given as argument
    # INCLUDES the level-0 as well. It means there will be num_levels-1 times
    # blurring and down_scaling.
    # INPUTS
    # @ img                 : 2D image (MxN)
    # @ sigma               : standard deviation of the Gaussian filter to be used at
    #                         all levels
    # @ filter_size         : the size of the filters
    # @ scale               : Downscaling factor (of type float, between 0-1)
    # OUTPUTS
    # @ gaussian_pyramid    : A list connatining images of pyramid. The first
    #                         element SHOULD be the image at the original scale
    #                         without any blurring.
   
    pass
