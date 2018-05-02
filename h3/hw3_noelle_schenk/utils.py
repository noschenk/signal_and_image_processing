from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import random
import os.path
from scipy.misc import imread


##############################################################################
#                        Functions for you to complete                       #
##############################################################################

################
# EXERCISE 2.1 #
################
# orig = im_array.copy()
# texture = np.copy(texture_img)
# patch_half_size = 3

# # acess a random edge pixel, save the current patch and mask
# a = np.where(find_edge(fill_region))
# # im_array[a[0][1], a[1][1]] = [0, 0, 255]
# patch = im_array[(a[0][1] - patch_half_size):(a[0][1]+ patch_half_size + 1), (a[1][1] -patch_half_size):(a[1][1] + patch_half_size + 1)]
# # plt.imshow(patch)
# mask = fill_region[(a[0][1] - patch_half_size):(a[0][1]+ patch_half_size + 1), (a[1][1] -patch_half_size):(a[1][1] + patch_half_size + 1)]
# mask[mask == 255] = 1
# mask = np.float32(mask)

def compute_ssd(patch, mask, texture, patch_half_size):
    # For all possible locations of patch in texture_img, computes sum square
    # difference for all pixels where mask = 0
    #
    # Inputs:
    #   patch: numpy array of size (2 * patch_half_size + 1, 2 * patch_half_size + 1, 3)
    #   mask: numpy array of size (2 * patch_half_size + 1, 2 * patch_half_size + 1)
    #   texture: numpy array of size (tex_rows, tex_cols, 3)
    #
    # Outputs:
    #   ssd: numpy array of size (tex_rows - 2 * patch_half_size, tex_cols - 2 * patch_half_size)
    patch_rows, patch_cols = np.shape(patch)[0:2]
    if (patch_rows != 2 * patch_half_size + 1 and patch_cols != 2 * patch_half_size + 1):
        plt.subplot(1, 3, 3)
        plt.imshow(im_filled)
        plt.title('Filled Image')
        plt.show()
        assert patch_rows == 2 * patch_half_size + 1 and patch_cols == 2 * patch_half_size + 1, "patch size and patch_half_size do not match."
    tex_rows, tex_cols = np.shape(texture)[0:2]
    ssd_rows = tex_rows - 2 * patch_half_size
    ssd_cols = tex_cols - 2 * patch_half_size
    ssd = np.zeros((ssd_rows, ssd_cols))
    for ind, value in np.ndenumerate(ssd):
        # print(ind, value)
        # take according pixel as the central pixel of the patch and find the according piece of the "texture" image
        # tex_center = (ind[0] + patch_half_size, ind[1] + patch_half_size)
        from_tex = texture[(ind[0]):(ind[0] + 2 * patch_half_size + 1), (ind[1]):(ind[1] + 2 * patch_half_size + 1)]
        # compare it to patch and calculate ssd (among all 3 dimensions) for values that are not 0 only,
        # save calculated ssd in ssd array
        ssd[ind] = np.sum((patch[mask != 1]-from_tex[mask != 1])**2)
    return ssd

# ssd = compute_ssd(patch, mask, texture, 3)
# plt.imshow(ssd)

################
# EXERCISE 2.2 #
################
# img = im_mask_fill.copy()
# iMatchCenter = np.where(ssd == np.max(ssd))[0] - patch_half_size
# jMatchCenter = np.where(ssd == np.max(ssd))[1] - patch_half_size
# iPatchCenter = a[0][1]
# jPatchCenter = a[1][1]
#
# # texture[iMatchCenter, jMatchCenter, : ] = [0,0,255]
# plt.imshow(texture)
#
# plt.imshow(img)
# img[iPatchCenter, jPatchCenter, :] = [0,0,255]


def copy_patch(img, mask, texture, iPatchCenter, jPatchCenter, iMatchCenter, jMatchCenter, patch_half_size):
    # Copies the patch of size (2 * patch_half_size + 1, 2 * patch_half_size + 1)
    # centered on (iMatchCenter, jMatchCenter) in texture into the image
    # img with center coordinates (iPatchCenter, jPatchCenter) for each
    # pixel where mask = 1.
    #
    # Inputs:
    #   img: ndarray of size (im_rows, im_cols, 3)
    #   mask: numpy array of size (2 * patch_half_size + 1, 2 * patch_half_size + 1)
    #   texture: numpy array of size (tex_rows, tex_cols, 3)
    #   iPatchCenter, jPatchCenter, iMatchCenter, jMatchCenter, patch_half_size: integers
    #
    # Outputs:
    #   res: ndarray of size (im_rows, im_cols, 3)

    patchSize = 2 * patch_half_size + 1
    iPatchTopLeft = iPatchCenter - patch_half_size
    jPatchTopLeft = jPatchCenter - patch_half_size
    iMatchTopLeft = iMatchCenter - patch_half_size
    jMatchTopLeft = jMatchCenter - patch_half_size
    res = img.copy()
    for i in range(patchSize):
        for j in range(patchSize):
            # print((i,j))
            # check if mask is 1 at the given place, if not we can go on to next iteration
            # if yes, the value of texture at the given place is filled in.
            if mask[i,j] == 1 :
                res[iPatchTopLeft + i, jPatchTopLeft + j] = texture[iMatchTopLeft + i, jMatchTopLeft + j]
    return res
# res = copy_patch(img, mask, texture, iPatchCenter, jPatchCenter, iMatchCenter, jMatchCenter, 3)
# plt.imshow(res)
# plt.imshow(img)


def find_edge(mask):
    # Returns the edges of a binary mask image.
    # The result edge_mask is a binary image highlighting the edges
    [cols, rows] = np.shape(mask)
    edge_mask = np.zeros(np.shape(mask))
    for y in range(rows):
        for x in range(cols):
            if (mask[x, y]):
                if (mask[x - 1, y] == 0 or mask[x + 1, y] == 0 or mask[x, y - 1] == 0 or mask[x, y + 1] == 0):
                    edge_mask[x, y] = 1
    return edge_mask
