import numpy as np
from collections import defaultdict
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sbn

import gower
from joblib import Parallel, delayed
from scipy.spatial import distance

import sys
import os.path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from error_control import control_inputs


# Functions that allow the display of the cost matrix together with the
# path followed to obtain the DTW distance value.

# Function that allows to obtain the path, that is, the route to
# reach the DTW distance value.
def get_path(x, y, D, path):
	
    i, j = len(x), len(y)
    while not (i == j == 0):
        # Add indexes to the path
        path.append((i - 1, j - 1))
        # Update indices
        i, j = D[i, j][1], D[i, j][2]
    # Flip the path to get it in the right order.
    return path.reverse()


# Function to obtain the cost matrix. 
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


# Function to paint the cost matrix.
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



# Function for the calculation of the independent DTW distance.
def dtw_ind(x, y, dist, dtw_distance=0, get_visualization=False):
	
    dim_m = x.shape[1]
    arr_D = []
    for index_m in range(dim_m):
        x_aux = x[:, index_m]
        y_aux = y[:, index_m]
        iter_object = [(i + 1, j + 1) for i in range(len(x)) for j in range(len(y))]
        # Starting conditions:
        # 1) D(0,0) = 0
        # 2) D(i,0) = inf
        # 3) D(0,j) = inf
        D = defaultdict(lambda: (float('inf'),))
        D[0, 0] = (0, 0, 0)
        if dist == "gower":
            for i, j in iter_object:
                # Given the function distance as input parameter the distance between x and y is calculated.
                # In case of using gower distance we fit the data.
                df = pd.DataFrame(np.array([x_aux, y_aux]))
                dt = gower.gower_matrix(df)[1][0]
                D[i, j] = min((D[i - 1, j][0] + dt, i - 1, j),
                              (D[i, j - 1][0] + dt, i, j - 1),
                              (D[i - 1, j - 1][0] + dt, i - 1, j - 1))
        else:
            for i, j in iter_object:
                # Given the function distance as input parameter the distance between x and y is calculated.
                # In case of using gower distance we fit the data.
                dt = dist(x_aux, y_aux)
                D[i, j] = min((D[i - 1, j][0] + dt, i - 1, j),
                              (D[i, j - 1][0] + dt, i, j - 1),
                              (D[i - 1, j - 1][0] + dt, i - 1, j - 1))
        if get_visualization:
            arr_D.append(D)

        dtw_distance += D[len(x_aux), len(y_aux)][0]

    return dtw_distance, arr_D
    

# Function for the calculation of the dependent DTW distance.
def dtw_dep(x, y, dist, mult_UTS=False):
    iter_object = [(i + 1, j + 1) for i in range(len(x)) for j in range(len(y))]
    # Starting conditions:
    # 1) D(0,0) = 0
    # 2) D(i,0) = inf
    # 3) D(0,j) = inf
    D = defaultdict(lambda: (float('inf'),))
    D[0, 0] = (0, 0, 0)
    if dist == "gower":
        for i, j in iter_object:
            # Given the function distance as input parameter the distance between x and y is calculated.
            # In case of using gower distance we fit the data.
            df = pd.DataFrame(np.array([x[i - 1], y[j - 1]]))
            dt = gower.gower_matrix(df)[1][0]
            D[i, j] = min((D[i - 1, j][0] + dt, i - 1, j),
                          (D[i, j - 1][0] + dt, i, j - 1),
                          (D[i - 1, j - 1][0] + dt, i - 1, j - 1))
    else:
        for i, j in iter_object:
            # Given the function distance as input parameter the distance between x and y is calculated.
            dt = dist(x[i - 1], y[j - 1])
            D[i, j] = min((D[i - 1, j][0] + dt, i - 1, j),
                          (D[i, j - 1][0] + dt, i, j - 1),
                          (D[i - 1, j - 1][0] + dt, i - 1, j - 1))

    if mult_UTS:
        return D[len(x), len(y)][0]
    else:
        return D[len(x), len(y)][0], D



def dtw(x, y=None, type_dtw="d", dist=distance.euclidean, MTS=False, get_visualization=False, check_errors=False, n_threads=-1, DTW_to_kernel=False, sigma_kernel=1):

    if check_errors:
        x, y = control_inputs(x, y, type_dtw, MTS)
    
    if MTS:        
        if type_dtw == "i":
            dtw_distance, D = dtw_ind(x, y, dist, get_visualization=get_visualization)
        else:
            dtw_distance, D = dtw_dep(x, y, dist)
            print(dtw_distance)
    else:
        # In case of having N UTS. We parallelize
        ## Data matrix (UTS) introduced in dataframe format
        if isinstance(x, pd.DataFrame):
            if y is None:
                y = x.copy()
            dtw_matrix_train = Parallel(n_jobs=n_threads)(
                delayed(dtw_dep)(x.loc[index_1,:].values, y.loc[index_2, :], dist, mult_UTS=True)
                for index_1 in range(x.shape[0]) 
                for index_2 in range(y.shape[0])
            )
            dtw_distance = np.array(dtw_matrix_train).reshape((len(x), len(y)))
            if DTW_to_kernel:
                return transform_DTW_to_kernel(dtw_distance, sigma_kernel)
        # Data matrix (UTS) introduced in array format
        else:
            if np.asanyarray(x, dtype='float').ndim > 1:
                if y == None:
                    y = x
                dtw_matrix_train = Parallel(n_jobs=n_threads)(
                    delayed(dtw_dep)(x[index_1], y[index_2], dist, mult_UTS=True)
                    for index_1 in range(len(x)) 
                    for index_2 in range(len(y))
                )
                dtw_distance = np.array(dtw_matrix_train).reshape((len(x), len(y)))
                if DTW_to_kernel:
                    return transform_DTW_to_kernel(dtw_distance, sigma_kernel)
            # In case of having 2 UTS.
            else:
                if y is None:
                    sys.stderr.write("You need introduce a vector -y")
                    sys.exit(0)
                dtw_distance, D = dtw_dep(x, y, dist)

                
    if get_visualization:
        if type_dtw == "i":
            for i in range(len(D)):
                path = []
                get_path(x, y, D[i], path)
                plot_cost_matrix(path, get_cost_matrix(D[i]))
        else:
            path = []
            get_path(x, y, D, path)
            cost_matrix = get_cost_matrix(D)
            plot_cost_matrix(path, cost_matrix)

    return dtw_distance


# We transform the DTW matrix to an exponential kernel. 
def transform_DTW_to_kernel(data, sigma_kernel):
	
	return np.exp(-data/(2*sigma_kernel**2))
	
	
# Function to obtain the calculation of the DTW distance at a high level. Parallelization is included.
def dtw_tensor_3d(X_1, X_2, input_obj):

    dtw_matrix_train = Parallel(n_jobs=input_obj.n_threads)(
        delayed(dtw)(X_1[i], y=X_2[j], type_dtw=input_obj.type_dtw, dist=input_obj.distance,
                     MTS=input_obj.MTS, get_visualization=input_obj.visualization, check_errors=input_obj.check_errors)
        for i in range(X_1.shape[0]) 
        for j in range(X_2.shape[0])
    )
    
    data = np.array(dtw_matrix_train).reshape((X_1.shape[0], X_2.shape[0]))

    if input_obj.DTW_to_kernel:
        return transform_DTW_to_kernel(data, input_obj.sigma_kernel)

    return data
