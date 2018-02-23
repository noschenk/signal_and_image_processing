##############################
#
# HW0 noelle schenk
#
#############################
# load modules
import numpy as np
import matplotlib.pyplot as plt

# Ex 0.1 ------------------------------------------------------------------------------------
# You are given a function f (x) = x+ x+ x 1 2 +10 sin(x). Create a function ex0(a,b,c)
# that plots f (x) where x are c equally spaced numbers in the range of [a, b]. In the
# case of b < a, the function returns −1 and does not plot, otherwise it returns 0 and
# plots the function.


def f(x):
    x = x+np.sqrt(x)+(1/x**2)+10*np.sin(x)
    return x


def ex0(a, b, c):
    # test if b < a and return -1 if true
    if b < a:
        return -1
    else:
        # create 1D array of c equally spaced numbers between a and b.
        x = np.linspace(a, b, c)
        # calculate y, being f(x)
        y = f(x)
        # plot x and f(x)
        plt.plot(x, y)
        plt.show()
        return None

# ex0(0.1,30,100) # test the function for a < b
# ex0(10,1,30) # test the function for b < a
# plt.close()
# -------------------------------------------------------------------------------------------


# Ex 0.2 ------------------------------------------------------------------------------------


# 0.2.1
# Write a function create image that generates an image with the height/width
# dimensions of n x m with uniform randomly distributed black and white pixels.
# Add a single red pixel at a random location.
# black and white are max or min of the given image range. If my image values range from
# 0 to 1.


def create_image(n, m):
    # create an array of numbers wich are either 0 or 1. To be black and white and
    # not grayscaled, I need to have all 3 dimensions having the same number.
    # I create a random 1 dimensional array with 0 and 1 and copy it 3 times.
    img = np.random.randint(0, high=2, size=(n, m))
    # img is stacked on itself three times
    # the data type needs to be float32 so matplotlib can show it with plt.imshow(<img>)
    img1 = np.dstack((img, img, img)).astype(np.float32)
    # at a random location, a red pixel is inserted. That means, the first dimension is
    #    at maximum (1) and the other 2 are at 0. (r,b,g) = (1,0,0)
    img1[np.random.randint(n), np.random.randint(m), :] = [1, 0, 0]
    return img1

# test the function:
# plt.imshow(create_image(5,6))
# -------------------------------------


# 0.2.2
# Next, write a function find pixels that finds the indexes of pixels with the
# values pixel values in an image img


def find_pixels(pixel_values, img):
    pos = np.logical_and(img[:, :, 0] == pixel_values[0],
                         img[:, :, 1] == pixel_values[1])  # np.logical_and only takes 2 args
    pos = np.logical_and(pos, img[:, :, 2] == pixel_values[2])
    result = np.argwhere(pos)
    return result


# test the function : which pixel has/have values (1,0,0) ?
test = create_image(5, 6)
find_pixels((1, 0, 0), test)
plt.imshow(test)
plt.show()

# ------------------------------------


# 0.2.3
# new exercise