# File error control of input
import numpy as np
from scipy.spatial import distance
import io
from contextlib import redirect_stdout

def control_inputs(x, y, type_dtw, MTS, term_exec):

    if type_dtw == "i" and not MTS:
        raise ValueError('Get independent dtw distance only valid for MTS.')

    if term_exec:
        x = [i[0] for i in x]
        y = [i[0] for i in y]
        
    x = np.asanyarray(x, dtype='float')
    y = np.asanyarray(y, dtype='float')
    

    # Irregular multivariate time series are not allowed.
    #if x.ndim == y.ndim > 1 and x.shape[1] != y.shape[1]:
    #    raise ValueError('Irregular multivariate time series are not allowed.')

    if x.ndim == 1 and MTS:
        raise ValueError('The data entered are not Multivariate Time Series.')

    if x.ndim == y.ndim > 1 and not MTS:
        raise ValueError('Change the value of the MTS flag.')

    #return x, y



# Functions to obtain the possible distances to be managed. 

def is_distance_function(func, checker):
    """
    Function that allows to check if the distance passed is inside a library, in this case, inside scipy.distance.
    """
    with io.StringIO() as buf, redirect_stdout(buf):
        help(func)
        output = buf.getvalue()

    if output.split("\n")[0].find(checker) == -1:
        return False
    else:
        return True
        
        
def possible_distances():
    """
    Check that the parameter introduced by terminal associated to the distance is one of the possible parameters to use.

    :return: possible distances
    """
    possible_distance = []
    for i in range(len(dir(distance))):
        if(len(dir(distance)[i].split("_")) == 1) and not any(c.isupper() for c in dir(distance)[i]):
            if is_distance_function("scipy.spatial.distance."+dir(distance)[i],
            checker = "function " + dir(distance)[i] + " in scipy.spatial.distance"):
                possible_distance.append(dir(distance)[i])

    # I add as a possible distance measurement the gower distance. Gower distance allows the
    # calculation of distance between continuous and binary variables.
    possible_distance.append("gower")
    # I include the computation of norm 1 and norm 2 optimized in terms of computational time.
    possible_distance.append("norm1")
    possible_distance.append("norm2")
    possible_distance.append("square_euclidean_distance")
    
    return possible_distance
   
