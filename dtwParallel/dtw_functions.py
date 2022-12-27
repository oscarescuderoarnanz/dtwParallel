## Functions _njit_itakura_mask, itakura_mask, sakoe_chiba_mask and compute_mask
## have their origin in the tslearned project 
## (https://github.com/tslearn-team/tslearn/blob/main/tslearn/metrics/dtw_variants.py)
## released under a BSD 2-Clause license and with Copyright (c) 2017, Romain Tavenard
## We incorporate them into dtwParallel, meeting the conditions specified in the license
## as it can be seen in the LICENSE file of dtwParallel.

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

from numba import njit, prange

GLOBAL_CONSTRAINT_CODE = {None: 0, "": 0, "itakura": 1, "sakoe_chiba": 2}


def to_time_series(ts):
    ts_out = np.array(ts, copy=True)
    if ts_out.ndim <= 1:
        ts_out = ts_out.reshape((-1, 1))
    if ts_out.dtype != float:
        ts_out = ts_out.astype('float64')
    
    return ts_out


@njit()
def _njit_itakura_mask(sz1, sz2, max_slope=2.):
    """Compute the Itakura mask without checking that the constraints
    are feasible. In most cases, you should use itakura_mask instead.
    Parameters
    ----------
    sz1 : int
        The size of the first time series
    sz2 : int
        The size of the second time series.
    max_slope : float (default = 2)
        The maximum slope of the parallelogram.
    Returns
    -------
    mask : array, shape = (sz1, sz2)
        Itakura mask.
    """
    min_slope = 1 / float(max_slope)
    max_slope *= (float(sz1) / float(sz2))
    min_slope *= (float(sz1) / float(sz2))

    lower_bound = np.empty((2, sz2))
    lower_bound[0] = min_slope * np.arange(sz2)
    lower_bound[1] = ((sz1 - 1) - max_slope * (sz2 - 1)
                      + max_slope * np.arange(sz2))
    lower_bound_ = np.empty(sz2)
    for i in prange(sz2):
        lower_bound_[i] = max(round(lower_bound[0, i], 2),
                              round(lower_bound[1, i], 2))
    lower_bound_ = np.ceil(lower_bound_)

    upper_bound = np.empty((2, sz2))
    upper_bound[0] = max_slope * np.arange(sz2)
    upper_bound[1] = ((sz1 - 1) - min_slope * (sz2 - 1)
                      + min_slope * np.arange(sz2))
    upper_bound_ = np.empty(sz2)
    for i in prange(sz2):
        upper_bound_[i] = min(round(upper_bound[0, i], 2),
                              round(upper_bound[1, i], 2))
    upper_bound_ = np.floor(upper_bound_ + 1)

    mask = np.full((sz1, sz2), np.inf)
    for i in prange(sz2):
        mask[int(lower_bound_[i]):int(upper_bound_[i]), i] = 0.
    return mask


def itakura_mask(sz1, sz2, max_slope=2.):
    """Compute the Itakura mask.
    Parameters
    ----------
    sz1 : int
        The size of the first time series
    sz2 : int
        The size of the second time series.
    max_slope : float (default = 2)
        The maximum slope of the parallelogram.
    Returns
    -------
    mask : array, shape = (sz1, sz2)
        Itakura mask.
    Examples
    --------
    >>> itakura_mask(6, 6)
    array([[ 0., inf, inf, inf, inf, inf],
           [inf,  0.,  0., inf, inf, inf],
           [inf,  0.,  0.,  0., inf, inf],
           [inf, inf,  0.,  0.,  0., inf],
           [inf, inf, inf,  0.,  0., inf],
           [inf, inf, inf, inf, inf,  0.]])
    """
    mask = _njit_itakura_mask(sz1, sz2, max_slope=max_slope)

    # Post-check
    raise_warning = False
    for i in prange(sz1):
        if not np.any(np.isfinite(mask[i])):
            raise_warning = True
            break
    if not raise_warning:
        for j in prange(sz2):
            if not np.any(np.isfinite(mask[:, j])):
                raise_warning = True
                break
    if raise_warning:
        warnings.warn("'itakura_max_slope' constraint is unfeasible "
                      "(ie. leads to no admissible path) for the "
                      "provided time series sizes",
                      RuntimeWarning)

    return mask


@njit()
def sakoe_chiba_mask(sz1, sz2, radius=1):
    """Compute the Sakoe-Chiba mask.
    Parameters
    ----------
    sz1 : int
        The size of the first time series
    sz2 : int
        The size of the second time series.
    radius : int
        The radius of the band.
    Returns
    -------
    mask : array, shape = (sz1, sz2)
        Sakoe-Chiba mask.
    Examples
    --------
    >>> sakoe_chiba_mask(4, 4, 1)
    array([[ 0.,  0., inf, inf],
           [ 0.,  0.,  0., inf],
           [inf,  0.,  0.,  0.],
           [inf, inf,  0.,  0.]])
    >>> sakoe_chiba_mask(7, 3, 1)
    array([[ 0.,  0., inf],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [inf,  0.,  0.]])
    """
    mask = np.full((sz1, sz2), np.inf)
    if sz1 > sz2:
        width = sz1 - sz2 + radius
        for i in prange(sz2):
            lower = max(0, i - radius)
            upper = min(sz1, i + width) + 1
            mask[lower:upper, i] = 0.
    else:
        width = sz2 - sz1 + radius
        for i in prange(sz1):
            lower = max(0, i - radius)
            upper = min(sz2, i + width) + 1
            mask[i, lower:upper] = 0.
    return mask


def compute_mask(s1, s2, global_constraint=0,
                 sakoe_chiba_radius=None, itakura_max_slope=None):
    """Compute the mask (region constraint).
    Parameters
    ----------
    s1 : array
        A time series or integer.
    s2: array
        Another time series or integer.
    global_constraint : {0, 1, 2} (default: 0)
        Global constraint to restrict admissible paths for DTW:
        - "itakura" if 1
        - "sakoe_chiba" if 2
        - no constraint otherwise
    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        If None and `global_constraint` is set to 2 (sakoe-chiba), a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to 1 (itakura), a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    Returns
    -------
    mask : array
        Constraint region.
    """
    # The output mask will be of shape (sz1, sz2)
    if isinstance(s1, int) and isinstance(s2, int):
        sz1, sz2 = s1, s2
    else:
        sz1 = s1.shape[0]
        sz2 = s2.shape[0]
    if (global_constraint == 0 and sakoe_chiba_radius is not None
            and itakura_max_slope is not None):
        raise RuntimeWarning("global_constraint is not set for DTW, but both "
                             "sakoe_chiba_radius and itakura_max_slope are "
                             "set, hence global_constraint cannot be inferred "
                             "and no global constraint will be used.")
    if global_constraint == 2 or (global_constraint == 0
                                  and sakoe_chiba_radius is not None):
        if sakoe_chiba_radius is None:
            sakoe_chiba_radius = 1
        mask = sakoe_chiba_mask(sz1, sz2, radius=sakoe_chiba_radius)
    elif global_constraint == 1 or (global_constraint == 0
                                    and itakura_max_slope is not None):
        if itakura_max_slope is None:
            itakura_max_slope = 2.
        mask = itakura_mask(sz1, sz2, max_slope=itakura_max_slope)
    else:
        mask = np.zeros((sz1, sz2))
    return mask


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
def general_dtw_ind(type_distance, mask, len_ts1, len_ts2, ts1, ts2, cost_matrix):

    for i in range(len_ts1):
        for j in range(len_ts2):
            if np.isfinite(mask[i, j]):
                cost_matrix[i + 1, j + 1] = type_distance(ts1, ts2)
                cost_matrix[i + 1, j + 1] += min(cost_matrix[i, j + 1],
                                             cost_matrix[i + 1, j],
                                             cost_matrix[i, j])
    return cost_matrix



def dtw_ind(ts1, ts2, local_dissimilarity, mask, dtw_distance=0, get_visualization=False):
    
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
            cost_matrix = general_dtw_ind(eval(local_dissimilarity), mask, len_ts1, len_ts2, ts1_aux, ts2_aux, cost_matrix)

        elif local_dissimilarity == "gower":
            for i in range(len_ts1):
                for j in range(len_ts2):
                    if np.isfinite(mask[i, j]):
                        df = pd.DataFrame(np.array([ts1_aux, ts2_aux]))
                        cost_matrix[i + 1, j + 1] = gower.gower_matrix(df)[1][0]
                        cost_matrix[i + 1, j + 1] += min(cost_matrix[i, j + 1],
                                               cost_matrix[i + 1, j],
                                               cost_matrix[i, j])
        else:
            for i in range(len_ts1):
                for j in range(len_ts2):
                    if np.isfinite(mask[i, j]):
                        cost_matrix[i + 1, j + 1] = local_dissimilarity(ts1_aux, ts2_aux)
                        cost_matrix[i + 1, j + 1] += min(cost_matrix[i, j + 1],
                                                         cost_matrix[i + 1, j],
                                                         cost_matrix[i, j])

        if get_visualization:
            arr_cost_matrix.append(cost_matrix)
        

        dtw_distance += cost_matrix[-1,-1]

    return dtw_distance, arr_cost_matrix



@njit()
def general_dtw_dep(local_dissimilarity, mask, len_ts1, len_ts2, ts1, ts2, cost_matrix):

    for i in range(len_ts1):
        for j in range(len_ts2):
            if np.isfinite(mask[i, j]):
                cost_matrix[i + 1, j + 1] = local_dissimilarity(ts1[i], ts2[j])
                cost_matrix[i + 1, j + 1] += min(cost_matrix[i, j + 1],
                                             cost_matrix[i + 1, j],
                                             cost_matrix[i, j])
    return cost_matrix



def dtw_dep(ts1, ts2, local_dissimilarity, mask, mult_uts=False, regular_flag=0):

    len_ts1 = len(ts1)
    len_ts2 = len(ts2)

    cost_matrix = np.full((len_ts1+1, len_ts2+1), np.inf)
    cost_matrix[0, 0] = 0.

    if local_dissimilarity in ["norm1", "norm2", "square_euclidean_distance"]:
        ts1 = to_time_series(ts1)
        ts2 = to_time_series(ts2)
        #len_ts1 = ts1.shape[0]
        #len_ts2 = ts2.shape[0]
        cost_matrix = general_dtw_dep(eval(local_dissimilarity), mask, len_ts1, len_ts2, ts1, ts2, cost_matrix)

    elif local_dissimilarity == "gower":
        for i in range(len_ts1):
            for j in range(len_ts2):
                if np.isfinite(mask[i, j]):
                    df = pd.DataFrame(np.array([ts1[i], ts2[j]]))
                    cost_matrix[i + 1, j + 1] = gower.gower_matrix(df)[1][0]
                    cost_matrix[i + 1, j + 1] += min(cost_matrix[i, j + 1],
                                           cost_matrix[i + 1, j],
                                           cost_matrix[i, j])
    else:
        for i in range(len_ts1):
            for j in range(len_ts2):
                if np.isfinite(mask[i, j]):
                    cost_matrix[i + 1, j + 1] = local_dissimilarity(ts1[i], ts2[j])
                    cost_matrix[i + 1, j + 1] += min(cost_matrix[i, j + 1],
                                           cost_matrix[i + 1, j],
                                           cost_matrix[i, j])

    if mult_uts:
        return cost_matrix[-1,-1]

    # irregular time series
    if regular_flag != 0:
        return cost_matrix[-1,-1] / np.sqrt(len(ts1)*len(ts2)), cost_matrix

    return cost_matrix[-1,-1], cost_matrix



def process_irregular_ts_dtw_ind(ts1, ts2, regular_flag):
    """
    This function eliminates the values indicated in the flag, obtaining irregular multivariate time series. After this, we replicate the last value of the time series (TS) until we obtain a regular TS.

    Parameters
    ------------
    :param ts1: time serie 1
    :param ts2: time serie 2
    :param regular_flag: int value

    :return: time serie 1 and time serie 2
    """

    ts1 = ts1[0:len(np.unique(np.where(ts1 != regular_flag)[0]))]
    ts2 = ts2[0:len(np.unique(np.where(ts2 != regular_flag)[0]))]

    if ts1.shape[0] < ts2.shape[0]:
        ts1_aux = ts1.copy()
        if ts1.shape[0] > 1:
            ts1_aux = ts1[-1].reshape((1,-1))
        
        for _ in range(ts2.shape[0] - ts1.shape[0]):
            ts1 = np.concatenate((ts1, ts1_aux))

    elif ts2.shape[0] < ts1.shape[0]:
        ts2_aux = ts2.copy()
        if ts2.shape[0] > 1:
            ts2_aux = ts2[-1].reshape((1,-1))

        for _ in range(ts1.shape[0] - ts2.shape[0]):
            ts2 = np.concatenate((ts2, ts2_aux))

    return ts1, ts2



def get_mask(ts1, ts2, constrained_path_search, sakoe_chiba_radius, itakura_max_slope):
    """
    Compute the mask (region constraint)

    Parameters
    ------------
    :param ts1: A time series or integer
    :param ts2: Another time series or integer
    :param constrained_path_search: type constraint (None, sakoe-chiba o itakura)
    :param sakoe_chiba_radius: int or None
    :param itakura_max_slope: float or None

    :return: array
        Constraint region
    """

    ts1 = to_time_series(ts1)
    ts2 = to_time_series(ts2)

    mask = compute_mask(
        ts1, ts2,
        GLOBAL_CONSTRAINT_CODE[constrained_path_search],
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope)

    return mask



def dtw(ts1, ts2=None, type_dtw="d", constrained_path_search=None, local_dissimilarity=distance.euclidean, MTS=False, get_visualization=False, check_errors=False, regular_flag=0, n_threads=-1, dtw_to_kernel=False, sigma_kernel=1, itakura_max_slope=None, sakoe_chiba_radius=None, term_exec=False):

    if check_errors:
        control_inputs(ts1, ts2, type_dtw, MTS, term_exec)

    mask = get_mask(ts1, ts2, constrained_path_search, sakoe_chiba_radius, itakura_max_slope)
    
    if MTS:        
        if type_dtw == "i":

            if regular_flag != 0:
                ts1, ts2 = process_irregular_ts_dtw_ind(ts1, ts2, regular_flag)

            dtw_distance, cost_matrix = dtw_ind(ts1, ts2, local_dissimilarity, mask,  get_visualization=get_visualization)
        else:
            if regular_flag != 0:
                ts1 = ts1[0:len(np.unique(np.where(ts1 != regular_flag)[0]))]
                ts2 = ts2[0:len(np.unique(np.where(ts2 != regular_flag)[0]))]

            dtw_distance, cost_matrix = dtw_dep(ts1, ts2, local_dissimilarity, mask, regular_flag=regular_flag)
    else:
        # In case of having N UTS. We parallelize
        ## Data matrix (UTS) introduced in dataframe format
        if isinstance(ts1, pd.DataFrame) and ts1.shape[0] > 1:
            if ts2 is None:
                ts2 = ts1.copy()

            dtw_matrix_train = Parallel(n_jobs=n_threads)(
                delayed(dtw_dep)(ts1.loc[index_1,:].values, ts2.loc[index_2, :], local_dissimilarity, mask, mult_UTS=True)
                for index_1 in range(ts1.shape[0]) 
                for index_2 in range(ts2.shape[0])
            )

            dtw_distance = np.array(dtw_matrix_train).reshape((len(ts1), len(ts2)))

            if dtw_to_kernel:
                return dtw_distance, transform_dtw_to_kernel(dtw_distance, sigma_kernel)

        # In case we have a unidimensional UTS with dataframe format.
        elif isinstance(ts1, pd.DataFrame) and ts1.shape[0] == 1:
            dtw_distance, cost_matrix = dtw_dep(ts1, ts2, local_dissimilarity, mask)
        
        # If we hace a data matrix (UTS) introduced in array format with N UTS >= 2.
        else:
            if np.asanyarray(ts1, dtype='float').ndim > 1 and not(isinstance(ts1, pd.DataFrame)) and not term_exec:
                if ts2 is None:
                    ts2 = ts1
                
                len_ts1 = len(ts1)
                len_ts2 = len(ts2)
                
                dtw_matrix_train = Parallel(n_jobs=n_threads)(
                    delayed(dtw_dep)(ts1[index_1], ts2[index_2], local_dissimilarity, mask, mult_UTS=True)
                    for index_1 in range(len_ts1) 
                    for index_2 in range(len_ts2)
                )

                dtw_distance = np.array(dtw_matrix_train).reshape((len_ts1, len_ts2))

                if dtw_to_kernel:
                    return dtw_distance, transform_dtw_to_kernel(dtw_distance, sigma_kernel)

            # In case of having 2 UTS.
            else:
                dtw_distance, cost_matrix = dtw_dep(ts1, ts2, local_dissimilarity, mask)


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


def transform_dtw_to_kernel(data, sigma_kernel):
    """
    We transform the DTW matrix to an exponential kernel.

    Parameters
    ------------
    :param data: numpy.ndarray
        DTW distance matrix
    :param sigma_kernel: float

    :return: numpy.ndarray
        Transformation of the DTW distance matrix to exponential kernel.
    """
	
    return np.exp(-data/(2*sigma_kernel**2))
	

def dtw_tensor_3d(mts1, mts2, input_obj):
    """
    Function to obtain the calculation of the DTW distance at a high level. Parallelization is included.

    Parameters
    ------------
    :param mts1: tensor of N MTS.
    :param mts2: Another tensor of N MTS.
    :param input_obj: object with parameters.

    :return: numpy.ndarray
        DTW matrix or matrix kernel.
    """

    dtw_matrix_train = Parallel(n_jobs=input_obj.n_threads)(
        delayed(dtw)(mts1[i], mts2[j], type_dtw=input_obj.type_dtw, constrained_path_search=input_obj.constrained_path_search, local_dissimilarity=input_obj.local_dissimilarity,
                      MTS=input_obj.MTS, get_visualization=input_obj.visualization,
                      check_errors=input_obj.check_errors, regular_flag=input_obj.regular_flag,
                      itakura_max_slope=input_obj.itakura_max_slope, sakoe_chiba_radius=input_obj.sakoe_chiba_radius)
        for i in range(mts1.shape[0])
        for j in range(mts2.shape[0])
    )
    
    data = np.array(dtw_matrix_train).reshape((mts1.shape[0], mts2.shape[0]))

    if input_obj.dtw_to_kernel:
        return data, transform_dtw_to_kernel(data, input_obj.sigma_kernel)

    return data
