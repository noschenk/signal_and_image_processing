##############################
#
# HW0 noelle schenk
#
#############################
# load modules
import numpy as np
import matplotlib.pyplot as plt
# import pdb; pdb.set_trace()  # python debugger

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
# image img, compute euclidean distance of each white pixel from the red pixel without the use of any for loop.
plt.imshow(test)  # the image to work with.


def compute_distances(test):
    # store the coordinates of white pixels
    whitepix = find_pixels((1, 1, 1), test)
    # store the coordinates of the red pixel.
    redpix = find_pixels((1, 0, 0), test)
    # calculate euclidean distance
    dist = (whitepix - redpix) ** 2  # numpy automatically acts as if redpix had the same dimensions as whitepix!
    dist = np.sum(dist, axis=1)  # sum square of coordinates together
    dist = np.sqrt(dist)
    return dist

# test the function with the image 'test'
dist = compute_distances(test)


# ------------------------------------
# 0.2.4
# Display the computed distance vector dist in a histogram (with 100 bins),
# compute the mean, standard deviation and the median. Display the values as
# a title above the plot.


def visualize_results(dist):
    """outputs histogram of the computed distance vector dist"""
    plt.hist(dist, bins=100)
    plt.title('mean = %f, std = %f, median = %f' % (np.mean(dist), np.std(dist), np.median(dist)))
    # %f for a float, the order is the same in the string and how variables are given
    plt.show()

# testing the function
visualize_results(dist)
# -------------------------------------------------------------------------------------------


# Ex 0.3 ------------------------------------------------------------------------------------

# import the image as stop
stop = plt.imread('h0/stopturnsigns.jpg')
# create copy of the image (if not, the image can not be changed...)
stop = np.copy(stop)

# find t_min and t_max that fulfill the desired things
t_min = [210, 0, 0]
t_max = [255, 85, 85]
# pixels between t_min and t_max are red
m = np.logical_and(stop[:, :, :] >= t_min, stop[:, :, :] <= t_max)
# pixels must be in the range at the same time at all 3 color channels
m = np.logical_and(m[:, :, 0], m[:, :, 1], m[:, :, 2])
# copy[np.logical_not(m)] = [0, 0, 255]
m = np.logical_not(m)
m[140:, :] = True
stop[m] = [0, 0, 255]
plt.imshow(stop)
plt.show()

# # other way of doing it, but without t_min and t_max
# # make the whole copied image blue
# copy[:,:,:] = [0, 0, 255]
# # add the stop sign back in
# copy[30:140, 80:190, :] = stop[30:140, 80:190, :]
# # setting values to blue which are 'not red enough'
# test = np.logical_or(copy[:, :, 0] < 230, copy[:, :, 1] > 140, copy[:, :, 2] > 140)
# copy[test] = [0,0,255]
# -------------------------------------------------------------------------------------------


# Ex 0.4 ------------------------------------------------------------------------------------
# Considering two 1 dimensional arrays, calculate the mean square error between them.
# Check your function for arrays of known difference
# ------------------------------------
# 0.4.1
# 2 arrays x, y of length 100 with random values
x = np.random.rand(100)
y = np.random.rand(100)

# 0.4.2


def mse(a, b):
    """calculation of mean square error"""
    res = (1 / a.shape[0]) * np.sum((a - b)**2)
    return res

# 0.4.3
mse(x, y)

# 0.4.4
mse(x, x)

# 0.4.5
# offset = verschiebung
# expectation : I expect that mean square errors differ by 2, as all values are offset by 2.
mse(x, x+2)
# observation : the mse is 4, the square of 2.
# test : if i calculate mse(x, x+3) the mse should be 9, if i use offset of 4, the mse is 16, etc.
mse(x, x+3), mse(x, x+4), mse(x, x+12)
# conclusion : it seems to be correct that it's the square of the offset.

# 0.4.6
# take the function f(x) from exercise 1.
x = np.arange(201)[1:]
plt.plot(x, f(x))
plt.plot(x, 1.2*x)
# visual check : it seems to be a good approximation for low values (until maybe x=100) and then the linear
# approx. starts to be too high.
# As it makes things a lot easier to work with a linear approximation, it could be accepted depending on the task.

# 0.4.7
mse(f(x), 1.2*x) # = 235.19621347180933

# 0.4.8
# define the range
a = np.arange(0, 5, step=0.1)
# calculate mse for each value of a
mse_range = np.array([])
for i in a:
    new = np.array(mse(f(x), i*x))
    mse_range = np.append(mse_range, new)
# plot mse values for the given values of a
plt.plot(a, mse_range)
print('the minimum MSE value is %f and the optimal value for a is %f'
      % (np.min(mse_range), a[np.argmin(mse_range)]))
