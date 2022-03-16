import numpy as np
from collections import defaultdict
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sbn

import gower


def plot_cost_matrix(warp_path, cost):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax = sbn.heatmap(cost, annot=True, square=True, linewidths=0.1, cmap="YlGnBu", ax=ax)

    # Get the warp path in x and y directions
    path_x = [p[0] for p in warp_path]
    path_y = [p[1] for p in warp_path]

    # Align the path from the center of each cell
    path_xx = [x + 0.5 for x in path_x]
    path_yy = [y + 0.5 for y in path_y]

    ax.plot(path_xx, path_yy, color='blue', linewidth=3, alpha=0.2)


def control_inputs(x, y, type_dtw, MTS):
    if type_dtw == "i" and not MTS:
        raise ValueError('Get independent dtw distance only valid for MTS.')

    x = np.asanyarray(x, dtype='float')
    y = np.asanyarray(y, dtype='float')


    # Irregular multivariate time series are not allowed.
    if x.ndim == y.ndim > 1 and x.shape[1] != y.shape[1]:
        raise ValueError('Irregular multivariate time series are not allowed.')

    if x.ndim == 1 and MTS:
        raise ValueError('The data entered are not Multivariate Time Series.')

    if x.ndim == y.ndim > 1 and not MTS:
        raise ValueError('Change the value of the MTS flag.')

    return x, y


def dtw_ind(x, y, dist, dtw_distance=0):
    dim_m = x.shape[1]
    for index_m in range(dim_m):
        x_aux = x[:, index_m]
        y_aux = y[:, index_m]
        iter_object = [(1, 1) for i in range(x_aux.ndim) for j in range(y_aux.ndim)]
        # Starting conditions:
        # 1) D(0,0) = 0
        # 2) D(i,0) = inf
        # 3) D(0,j) = inf
        D = defaultdict(lambda: (float('inf'),))
        D[0, 0] = (0, 0, 0)
        for i, j in iter_object:
            # Given the function distance as input parameter the distance between x and y is calculated.
            # In case of using gower distance we fit the data.
            if dist == gower.gower_matrix:
                df = pd.DataFrame(np.array([x_aux, y_aux]))
                dt = dist(df)[1][0]
            else:
                dt = dist(x_aux, y_aux)
            D[i, j] = min((D[i - 1, j][0] + dt, i - 1, j),
                          (D[i, j - 1][0] + dt, i, j - 1),
                          (D[i - 1, j - 1][0] + dt, i - 1, j - 1))

        dtw_distance += D[x_aux.ndim, y_aux.ndim][0]

    return dtw_distance


def dtw_dist(x, y, dist):
    iter_object = [(i + 1, j + 1) for i in range(len(x)) for j in range(len(y))]
    # Starting conditions:
    # 1) D(0,0) = 0
    # 2) D(i,0) = inf
    # 3) D(0,j) = inf
    D = defaultdict(lambda: (float('inf'),))
    D[0, 0] = (0, 0, 0)
    dist_matrix = np.zeros((len(x), len(y)))
    for i, j in iter_object:
        # Given the function distance as input parameter the distance between x and y is calculated.
        # In case of using gower distance we fit the data.
        if dist == gower.gower_matrix:
            df = pd.DataFrame(np.array([x[i - 1], y[j - 1]]))
            dt = dist(df)[1][0]
        else:
            dt = dist(x[i - 1], y[j - 1])
        dist_matrix[i - 1, j - 1] = dt
        D[i, j] = min((D[i - 1, j][0] + dt, i - 1, j),
                      (D[i, j - 1][0] + dt, i, j - 1),
                      (D[i - 1, j - 1][0] + dt, i - 1, j - 1))

    return D[len(x), len(y)][0], D


def get_path(x, y, D, path):
    i, j = len(x), len(y)
    while not (i == j == 0):
        # Add indexes to the path
        path.append((i - 1, j - 1))
        # Update indices
        i, j = D[i, j][1], D[i, j][2]
    # Flip the path to get it in the right order.
    return path.reverse()


def get_cost_matrix(D):
    positions_object = list(D)
    index = 0
    length_rows, length_columns = list(D.keys())[-1][0] + 1, list(D.keys())[-1][1] + 1
    cost_matrix = np.zeros((length_rows, length_columns))
    for i in range(length_rows):
        for j in range(length_columns):
            cost_matrix[positions_object[index]] = list(D.values())[index][0]
            index += 1

    arr = np.delete(cost_matrix, np.s_[0], 1)
    return np.delete(arr, np.s_[0], 0)


def dtw(x, y, type_dtw, dist, MTS=False, get_visualization=False):
    x, y = control_inputs(x, y, type_dtw, MTS)

    if MTS:
        if type_dtw == "i":
            dtw_distance = dtw_ind(x, y, dist)
        else:
            dtw_distance, _ = dtw_dist(x, y, dist)
    else:
        dtw_distance, D = dtw_dist(x, y, dist)

    if get_visualization:
        if type_dtw != "i" and not MTS:
            path = []
            get_path(x, y, D, path)
            cost_matrix = get_cost_matrix(D)
            plot_cost_matrix(path, cost_matrix)
        else:
            raise ValueError('Display not allowed. Only univariate case.')

    return dtw_distance
