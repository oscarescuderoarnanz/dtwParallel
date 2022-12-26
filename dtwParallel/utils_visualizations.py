import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
from numba import njit

# Functions that allow to visualize the cost matrix together with 
# the path followed to obtain the DTW distance value. In addition, 
# a new function has been implemented to visualize the alignment 
# between time series. 


@njit()
def get_path(cost_matrix):
    """
    Function that allows to obtain the path, that is, the route to reach the DTW distance value.

    Parameters
    ------------
    :param cost_matrix: cost matrix

    :return: list
        Optimal path
    """
    path = [(cost_matrix.shape[0] - 1, cost_matrix.shape[1] - 1)]
    while path[-1] != (0, 0):
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = np.array([cost_matrix[i - 1][j - 1],
                            cost_matrix[i - 1][j],
                            cost_matrix[i][j - 1]])
            argmin = np.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))

    return path[::-1][1:]


 
def plot_cost_matrix(warp_path, cost):
    """
    Function to paint the cost matrix.

    Parameters
    ------------
    :param warp_path: list
    :param cost: numpy.ndarray

    :return: non return
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax = sbn.heatmap(cost[1:,1:], annot=True, square=True, linewidths=0.1, cmap="YlGnBu", ax=ax)

    # Get the warp path in x and y directions
    path_x = [p[1] for p in warp_path]
    path_y = [p[0] for p in warp_path]

    # Align the path from the center of each cell
    path_xx = [x - 0.5 for x in path_x]
    path_yy = [y - 0.5 for y in path_y]

    ax.plot(path_xx, path_yy, color='blue', linewidth=3, alpha=0.2)


def plot_alignment(x, y, warp_path):
    """
    Function to paint the alignment between time series.

    Parameters
    ------------
    :param x: list
    :param y: list
    :param warp_path: list

    :return: non return
    """
    linewidths =[3.5, 3.5, 0.5]

    fig, ax = plt.subplots(figsize=(10, 8))

    # ASSESS WHETHER OR NOT TO REMOVE
    fig.patch.set_visible(False)
    ax.axis('off')

    # This prevents unexpected changes in the reference signal after the duplicate
    x_ref = np.copy(x)  
    # Set an offset for visualization
    x_ref += 2 * np.max(x)  

    xref = np.arange(len(x_ref))

    plt.plot(xref, x_ref, color="green", lw=linewidths[0], label="Time series 1")
    plt.plot(y, color="blue", lw=linewidths[1], label="Time series 2")

    [
        plt.plot(
            [warp_path[i][0], warp_path[i][1]],
            [x_ref[warp_path[i][0]], y[warp_path[i][1]]],
            color="k",
            lw=linewidths[2],
        )
        for i in range(len(warp_path)-1)
    ]

    plt.legend(fontsize=14)