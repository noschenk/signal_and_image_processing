import sys
import matplotlib
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'nearest'
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature, color


##############################################################################
#                        Functions to complete                               #
##############################################################################


################
# EXERCISE 1.1 #
################
# Implement a function find line model that fits a line y = mx + c through two given
# points (x 0 , y 0 ) and (x 1 , y 1 ).

# # to illustrate the results with small noise
# points = np.float_([[5, 0], [5, 6]])  # x = 5 case
# points = np.float_([[4, 1], [3, 1]])
# p = fit_line(points)
# def graph(formula, x_range):
#     x = np.array(range(x_range))
#     y = eval(formula)
#     plt.plot(x, y)
#     plt.show()
# graph(str(p[0]) + ' * x + ' + str(p[1]), 4)
# plt.title(str(p[0]) + ' * x + ' + str(p[1]))
# plt.suptitle('from points ' + str(points[0]) + ' and ' + str(points[1]))

def fit_line(points):
    # Fits a line y=m*x+c through two given points (x0,y0) and
    # (x1,y1). Returns the slope m and the y-intersect c of the line.
    #
    # Inputs:
    #   points: list with two 2D-points [[x0,y0], [x1,y1]]
    #           where x0,y0,x0,y1 are integers
    #
    # Outputs:
    #   m: the slope of the fitted line, integer
    #   c: the y-intersect of the fitted line, integers
    #
    # WARNING: vertical and horizontal lines should be treated differently
    #          here add some noise to avoid division by zero.
    #          You could use for example sys.float_info.epsilon

    # convert points to a numpy array with floats (if it is not already)
    points = np.float_(points)
    # convert all 0 values to sys.float_info.epsilon (in case there is just one 0)
    points[points == 0] = sys.float_info.epsilon
    # calculate the slope, treat horizontal and vertical lines differently
    if points[1, 0] - points[0, 0] == 0:
        print("vertical line with x = ", points[0, 0],
              "\n automatically added some noise with sys.float_info.epsilon")
        m = (points[1, 1] - points[0, 1]) / sys.float_info.epsilon
        # calculate c using the first point
        c = points[0, 1] - m * points[0, 0]
    elif (points[1, 1] - points[0, 1]) == 0:
        print("horizontal line with y = ", points[0, 1],
              "\n automatically added some noise with sys.float_info.epsilon")
        m = sys.float_info.epsilon / (points[1, 0] - points[0, 0])
        # calculate c using the first point
        c = points[0, 1] - m * points[0, 0]
    else:
        m = (points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0])
        # calculate c using the first point
        c = points[0, 1] - m * points[0, 0]

    # # calculate c using the first point
    # c = points[0, 1] - m * points[0, 0]
    # # calculate c using the second point
    # c_control = points[1, 1] - m * points[1, 0]
    # # control if c == c_control
    # # attention: it's comparing floats, so accept some errors when comparing!
    # if np.isclose(c, c_control):
    #     return m, c
    return m, c

################
# EXERCISE 1.2 #
################


def point_to_line_dist(m, c, x0, y0):
    # Returns the minimal distance between a given
    #  point (x0,y0)and a line y=m*x+c.
    #
    # Inputs:
    #   x0, y0: the coordinates of the points
    #   m, c: slope and intersect of the line
    #
    # Outputs:
    #   dist: the minimal distance between the point and the line.

    #
    # REPLACE THE FOLLOWING WITH YOUR CODE
    #
    dist = 0

    return dist


################
# EXERCISE 1.3 #
################

def edge_map(img):
    # Returns the edge map of a given image.
    #
    # Inputs:
    #   img: image of shape (n, m, 3) or (n, m)
    #
    # Outputs:
    #   edges: the edge map of image

    #
    # REPLACE THE FOLLOWING WITH YOUR CODE
    #
    edges = np.zeros(img.shape[0:2])
    #edges[np.random.randint(image.shape[0], size=100), np.random.randint(image.shape[1], size=100)] = 1

    return edges



##############################################################################
#                           Main script starts here                          #
##############################################################################

filename = 'synthetic.jpg'
#filename = 'bridge.jpg'
#filename = 'pool.jpg'
#filename = 'tennis.jpg'

image = plt.imread(filename)
edges = edge_map(image)

f1 = plt.figure()

plt.imshow(edges)
plt.title('edge map')

f1.show()

edge_pts = np.array(np.nonzero(edges), dtype=float).T
edge_pts_xy = edge_pts[:, ::-1]

ransac_iterations = 500
ransac_threshold = 2
n_samples = 2

ratio = 0

# perform RANSAC iterations
for it in range(ransac_iterations):

    # this shows progress
    sys.stdout.write('\r')
    sys.stdout.write('iteration {}/{}'.format(it+1, ransac_iterations))
    sys.stdout.flush()

    all_indices = np.arange(edge_pts.shape[0])
    np.random.shuffle(all_indices)

    indices_1 = all_indices[:n_samples]
    indices_2 = all_indices[n_samples:]

    maybe_points = edge_pts_xy[indices_1, :]
    test_points = edge_pts_xy[indices_2, :]

    # find a line model for these points
    m, c = fit_line(maybe_points)

    x_list = []
    y_list = []
    num = 0

    # find distance to the model for all testing points
    for ind in range(test_points.shape[0]):

        x0 = test_points[ind, 0]
        y0 = test_points[ind, 1]

        # distance from point to the model
        dist = point_to_line_dist(m, c, x0, y0)

        # check whether it's an inlier or not
        if dist < ransac_threshold:
            num += 1

    # in case a new model is better - cache it
    if num / float(n_samples) > ratio:
        ratio = num / float(n_samples)
        model_m = m
        model_c = c

x = np.arange(image.shape[1])
y = model_m * x + model_c

f2 = plt.figure()

if m != 0 or c != 0:
    plt.plot(x, y, 'r')
plt.imshow(image)

f2.show()

plt.show()
