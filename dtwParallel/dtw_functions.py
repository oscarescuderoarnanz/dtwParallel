import numpy as np
from collections import defaultdict
import pandas as pd

import gower
from joblib import Parallel, delayed
from scipy.spatial import distance

import sys
import os.path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from error_control import control_inputs
import utils_visualizations as uv

from numba import njit

from tslearn.metrics import dtw as dtw_tslearn

#import time 

#def timing_val(func):
#    def wrapper(*arg, **kw):
#        
#        t1 = time.time()
#        res = func(*arg, **kw)
#        t2 = time.time()
#        print(f'Function {func.__name__} Took {t2-t1:.4f} seconds')
#        return (t2 - t1)
#    return wrapper

# Function for the calculation of the independent DTW distance.

def to_time_series(ts):
    ts_out = np.array(ts, copy=True)
    if ts_out.ndim <= 1:
        ts_out = ts_out.reshape((-1, 1))
    if ts_out.dtype != float:
        ts_out = ts_out.astype('float64')
    
    return ts_out


@njit()
def norm2(s1, s2):
    dist = 0.
    for di in range(s1.shape[0]):
        diff = np.abs(s1[di] - s2[di])
        dist += diff * diff
    return np.sqrt(dist)


@njit()
def norm1(s1, s2):
    dist = 0.
    for di in range(s1.shape[0]):
        dist += np.abs(s1[di] - s2[di])
    return np.sqrt(dist)


@njit()
def square_euclidean_distance(s1, s2):
    dist = 0.
    for di in range(s1.shape[0]):
        diff = s1[di] - s2[di]
        dist += diff * diff
    return dist


@njit()
def general_dtw_ind(typeDistance, len_ts1, len_ts2, ts1, ts2, cost_matrix):

    for i in range(len_ts1):
        for j in range(len_ts2):
            cost_matrix[i + 1, j + 1] = typeDistance(ts1, ts2)
            cost_matrix[i + 1, j + 1] += min(cost_matrix[i, j + 1],
                                         cost_matrix[i + 1, j],
                                         cost_matrix[i, j])
    return cost_matrix


#@njit()
#def optimize_distances_dtw_ind(ts1, ts2, len_ts1, len_ts2, cost_matrix, local_dissimilarity):

    #if local_dissimilarity == "norm1":
    #    cost_matrix = general_dtw_ind(norm1, len_ts1, len_ts2, ts1, ts2, cost_matrix)
    #else:
    #    cost_matrix = general_dtw_ind(norm2, len_ts1, len_ts2, ts1, ts2, cost_matrix)

    #return cost_matrix


def dtw_ind(ts1, ts2, local_dissimilarity, dtw_distance=0, get_visualization=False, regular_flag=0):
    
    dim_m = ts1.shape[1]
    arr_cost_matrix = []
    len_ts1 = len(ts1)
    len_ts2 = len(ts2)

    for index_m in range(dim_m):
        ts1_aux = ts1[:, index_m]
        ts2_aux = ts2[:, index_m]

        cost_matrix = np.full((len_ts1+1, len_ts2+1), np.inf)
        cost_matrix[0, 0] = 0.

        if local_dissimilarity in ["norm1", "norm2", "square_euclidean_distance"]:
            cost_matrix = general_dtw_ind(eval(local_dissimilarity), len_ts1, len_ts2, ts1_aux, ts2_aux, cost_matrix)

        elif local_dissimilarity == "gower":
            for i in range(len_ts1):
                for j in range(len_ts2):
                    df = pd.DataFrame(np.array([ts1_aux, ts2_aux]))
                    cost_matrix[i + 1, j + 1] = gower.gower_matrix(df)[1][0]
                    cost_matrix[i + 1, j + 1] += min(cost_matrix[i, j + 1],
                                           cost_matrix[i + 1, j],
                                           cost_matrix[i, j])
        else:
            for i in range(len_ts1):
                for j in range(len_ts2):
                    cost_matrix[i + 1, j + 1] = local_dissimilarity(ts1_aux, ts2_aux)
                    cost_matrix[i + 1, j + 1] += min(cost_matrix[i, j + 1],
                                                     cost_matrix[i + 1, j],
                                                     cost_matrix[i, j])

        
        if get_visualization:
            arr_cost_matrix.append(cost_matrix)
        
        #if regular_flag != 0:
        #    dtw_distance += cost_matrix[-1,-1] / np.sqrt(len_ts1*len_ts2)
        #else:
        dtw_distance += cost_matrix[-1,-1]

    return dtw_distance, arr_cost_matrix



@njit()
def general_dtw_dep(local_dissimilarity, len_ts1, len_ts2, ts1, ts2, cost_matrix):

    for i in range(len_ts1):
        for j in range(len_ts2):
            cost_matrix[i + 1, j + 1] = local_dissimilarity(ts1[i], ts2[j])
            cost_matrix[i + 1, j + 1] += min(cost_matrix[i, j + 1],
                                         cost_matrix[i + 1, j],
                                         cost_matrix[i, j])
    return cost_matrix


#@njit()
#def optimize_distances_dtw_dep(ts1, ts2, cost_matrix, local_dissimilarity):

 #   len_ts1 = ts1.shape[0]
 #   len_ts2 = ts2.shape[0]

  #  general_dtw_dep(eval(local_dissimilarity), len_ts1, len_ts2, ts1, ts2, cost_matrix)
    #if local_dissimilarity == "norm1":
    #    cost_matrix = general_dtw_dep(norm1, len_ts1, len_ts2, ts1, ts2, cost_matrix)
    #else:
    #    cost_matrix = general_dtw_dep(norm2, len_ts1, len_ts2, ts1, ts2, cost_matrix)

   # return cost_matrix



#@timing_val
def dtw_dep(ts1, ts2, local_dissimilarity, mult_UTS=False, regular_flag=0):

    len_ts1 = len(ts1)
    len_ts2 = len(ts2)

    cost_matrix = np.full((len_ts1+1, len_ts2+1), np.inf)
    cost_matrix[0, 0] = 0.

    if local_dissimilarity in ["norm1", "norm2", "square_euclidean_distance"]:
        ts1 = to_time_series(ts1)
        ts2 = to_time_series(ts2)
        len_ts1 = ts1.shape[0]
        len_ts2 = ts2.shape[0]
        cost_matrix = general_dtw_dep(eval(local_dissimilarity), len_ts1, len_ts2, ts1, ts2, cost_matrix)

    elif local_dissimilarity == "gower":
        for i in range(len_ts1):
            for j in range(len_ts2):
                df = pd.DataFrame(np.array([ts1[i], ts2[j]]))
                cost_matrix[i + 1, j + 1] = gower.gower_matrix(df)[1][0]
                cost_matrix[i + 1, j + 1] += min(cost_matrix[i, j + 1],
                                       cost_matrix[i + 1, j],
                                       cost_matrix[i, j])
    else:
        for i in range(len_ts1):
            for j in range(len_ts2):
                cost_matrix[i + 1, j + 1] = local_dissimilarity(ts1[i], ts2[j])
                cost_matrix[i + 1, j + 1] += min(cost_matrix[i, j + 1],
                                       cost_matrix[i + 1, j],
                                       cost_matrix[i, j])


    
    if mult_UTS:
        return cost_matrix[-1,-1]

    # irregular time series
    if regular_flag != 0:
        return cost_matrix[-1,-1] / np.sqrt(len(ts1)*len(ts2)), cost_matrix

    return cost_matrix[-1,-1], cost_matrix
    #return np.sqrt(cost_matrix[-1,-1]) , cost_matrix



#def transform_pandas_to_ts(ts):
#    ts_out = np.array(ts, copy=True)
#    if ts_out.ndim >= 1:
#        ts_out = ts_out.reshape((1, -1))
#    if ts_out.dtype != float:
#        ts_out = ts_out.astype('float64')
#    
#    return ts_out


def process_irregular_ts_dtw_ind(ts1, ts2):

    ts1 = ts1[0:len(np.unique(np.where(ts1 != 666)[0]))]
    ts2 = ts2[0:len(np.unique(np.where(ts2 != 666)[0]))]

    if ts1.shape[0] < ts2.shape[0]:
        ts1_aux = ts1.copy()
        if ts1.shape[0] > 1:
            ts1_aux = ts1[-1].reshape((1,-1))
        
        for idx in range(ts2.shape[0]-ts1.shape[0]):
            ts1 = np.concatenate((ts1, ts1_aux))

    elif ts2.shape[0] < ts1.shape[0]:
        ts2_aux = ts2.copy()
        if ts2.shape[0] > 1:
            ts2_aux = ts2[-1].reshape((1,-1))

        for idx in range(ts1.shape[0]-ts2.shape[0]):
            ts2 = np.concatenate((ts2, ts2_aux))

    return ts1, ts2




#@timing_val
def dtw(ts1, ts2=None, type_dtw="d", local_dissimilarity=distance.euclidean, MTS=False, get_visualization=False, check_errors=False, regular_flag=0, n_threads=-1, DTW_to_kernel=False, sigma_kernel=1, itakura_max_slope=None, sakoe_chiba_radius=None):

    if type_dtw == "itakura":
        return dtw_tslearn(ts1, ts2, global_constraint="itakura", itakura_max_slope=itakura_max_slope)
    elif type_dtw == "sakoe_chiba":
        return dtw_tslearn(ts1, ts2, global_constraint="sakoe_chiba", sakoe_chiba_radius=sakoe_chiba_radius)

    if check_errors:
        x, y = control_inputs(ts1, ts2, type_dtw, MTS)
    
    if MTS:        
        if type_dtw == "i":

            if regular_flag != 0:
                ts1, ts2 = process_irregular_ts_dtw_ind(ts1, ts2)

            dtw_distance, cost_matrix = dtw_ind(ts1, ts2, local_dissimilarity, get_visualization=get_visualization, regular_flag=regular_flag)
        else:
            if regular_flag != 0:
                ts1 = ts1[0:len(np.unique(np.where(ts1 != 666)[0]))]
                ts2 = ts2[0:len(np.unique(np.where(ts2 != 666)[0]))]

            dtw_distance, cost_matrix = dtw_dep(ts1, ts2, local_dissimilarity, regular_flag=regular_flag)
    else:
        # In case of having N UTS. We parallelize
        ## Data matrix (UTS) introduced in dataframe format
        if isinstance(ts1, pd.DataFrame) and ts1.shape[1] > 1:
            if ts2 is None:
                ts2 = ts1.copy()

            dtw_matrix_train = Parallel(n_jobs=n_threads)(
                delayed(dtw_dep)(ts1.loc[index_1,:].values, ts2.loc[index_2, :], local_dissimilarity, mult_UTS=True)
                for index_1 in range(ts1.shape[0]) 
                for index_2 in range(ts2.shape[0])
            )

            dtw_distance = np.array(dtw_matrix_train).reshape((len(ts1), len(ts2)))

            if DTW_to_kernel:
                return dtw_distance, transform_DTW_to_kernel(dtw_distance, sigma_kernel)

        # Data matrix (UTS) introduced in array format
        else:
            if np.asanyarray(ts1, dtype='float').ndim > 1 and not(isinstance(ts1, pd.DataFrame)):
                if ts2 == None:
                    ts2 = ts1
                
                len_ts1 = len(ts1)
                len_ts2 = len(ts2)
                
                dtw_matrix_train = Parallel(n_jobs=n_threads)(
                    delayed(dtw_dep)(ts1[index_1], ts2[index_2], local_dissimilarity, mult_UTS=True)
                    for index_1 in range(len_ts1) 
                    for index_2 in range(len_ts2)
                )

                dtw_distance = np.array(dtw_matrix_train).reshape((len_ts1, len_ts2))

                if DTW_to_kernel:
                    return dtw_distance, transform_DTW_to_kernel(dtw_distance, sigma_kernel)

            # In case of having 2 UTS.
            else:
                # Esta parte del código debe ser arreglada!!! Considerar todo tipo de datos de entrada
                # REVISAR!!!!!!
                
                #if np.isnan(ts2):
                #    raise ValueError('You need introduce a UTS -y.')

                dtw_distance, cost_matrix = dtw_dep(ts1, ts2, local_dissimilarity)


    if get_visualization:
        if type_dtw == "i":
            for i in range(len(cost_matrix)):
                path = uv.get_path(cost_matrix[i])
                uv.plot_cost_matrix(path, cost_matrix[i])
                if not MTS:
                    uv.plot_alignment(ts1, ts2, path)
        else:
            path = uv.get_path(cost_matrix)
            uv.plot_cost_matrix(path, cost_matrix)
            if not MTS:
                uv.plot_alignment(ts1, ts2, path)

    return dtw_distance


# We transform the DTW matrix to an exponential kernel. 
def transform_DTW_to_kernel(data, sigma_kernel):
	
	return np.exp(-data/(2*sigma_kernel**2))
	

#@timing_val
# Function to obtain the calculation of the DTW distance at a high level. Parallelization is included.
def dtw_tensor_3d(X_1, X_2, input_obj):

    dtw_matrix_train = Parallel(n_jobs=input_obj.n_threads)(
        delayed(dtw)(X_1[i], X_2[j], type_dtw=input_obj.type_dtw, local_dissimilarity=input_obj.distance,
                      MTS=input_obj.MTS, get_visualization=input_obj.visualization, 
                      check_errors=input_obj.check_errors, regular_flag=input_obj.regular_flag,
                      itakura_max_slope=input_obj.itakura_max_slope, sakoe_chiba_radius=input_obj.sakoe_chiba_radius)
        for i in range(X_1.shape[0]) 
        for j in range(X_2.shape[0])
    )
    
    #dtw_matrix_train = np.zeros((X_1.shape[0], X_2.shape[0]))
    
    #for i in range(X_1.shape[0]): 
    #    for j in range(X_2.shape[0]):
    #        dtw_matrix_train[i,j] = dtw(X_1[i], X_2[j], type_dtw=input_obj.type_dtw, local_dissimilarity=input_obj.distance,
    #                  MTS=input_obj.MTS, get_visualization=input_obj.visualization, 
    #                  check_errors=input_obj.check_errors, regular_flag=input_obj.regular_flag,
    #                  itakura_max_slope=input_obj.itakura_max_slope, sakoe_chiba_radius=input_obj.sakoe_chiba_radius)
    #
    
    data = np.array(dtw_matrix_train).reshape((X_1.shape[0], X_2.shape[0]))
    #print(data.shape)

    if input_obj.DTW_to_kernel:
        return data, transform_DTW_to_kernel(data, input_obj.sigma_kernel)

    return data
