import pandas as pd
import numpy as np
import argparse
from io import StringIO
import configparser
import os.path
import sys
from scipy.spatial import distance
from error_control import possible_distances

DTW_DESC_MSG = \
"""
    Args:
        -x   Time series 1
        -y   Time series 2
        -t   Calculation type DTW (str)
        -d   Type of distance (function, norm1, norm2, square_euclidean_distance or gower)
        -ce  Check data for errors (bool)
        -of Output to File (bool)
        -nf Name of file (str)
        -vis Visualization of Time Series. only available when using dtwParallel with an API. (bool)
        -n Numero de hilos empleados para la paralelización (int)
        -k Transformación de la matriz de distancia a kernel (bool)
        -s Valor de sigma para la transformación a kernel exponencial aplicada (float)

    Others args:
        MTS  Bool value
    
    Optional arguments:
        -h, --help            show this help message and exit
        -v, --version         show version
        -l, --list            show available backends
    \n
"""

DTW_USAGE_MSG = \
"""
    %(prog)s [<args>] | --help | --version | --list \n   
""" \
+ str(DTW_DESC_MSG)

    
DTW_VERSION_MSG = \
"""
    %(prog)s 0.9.16
"""
    

def read_data(fname):
    return pd.read_csv(StringIO(fname.read()), header = None, sep=";")
    

def read_npy(fname):
    return np.load(fname.name)


class Input:
    def __init__(self):
        #Input.execute_configuration(self)
        config = configparser.ConfigParser()
        here = os.path.abspath(os.path.dirname(__file__))
        path_file_init = os.path.join(here, 'configuration.ini')

        # check if the path is to a valid file
        if not os.path.isfile(path_file_init):
            raise BadConfigError # not a standard python exception

        config.read(path_file_init)
        self.check_errors = config.getboolean('DEFAULT', 'check_errors')
        self.type_dtw = config.get('DEFAULT', 'type_dtw')
        self.MTS = config.getboolean('DEFAULT', 'MTS')
        self.regular_flag = config.getint('DEFAULT', 'regular_flag')
        self.n_threads = config.getint('DEFAULT', 'n_threads')
        
        # If the distance introduced is not correct, the execution is terminated.
        if self.check_errors:
            test_distance = possible_distances()
            if not config.get('DEFAULT', 'local_dissimilarity') in test_distance:
                raise ValueError('Distance introduced not allowed or incorrect.')
             
        #self.distance = eval("distance." + config.get('DEFAULT', 'distance'))
        self.local_dissimilarity = config.get('DEFAULT', 'local_dissimilarity')
        self.visualization = config.getboolean('DEFAULT', 'visualization')
        self.output_file = config.getboolean('DEFAULT', 'output_file')
        self.name_file = config['DEFAULT']['name_file']
        self.DTW_to_kernel = config.getboolean('DEFAULT', 'DTW_to_kernel')
        self.sigma_kernel = config.getint('DEFAULT', 'sigma_kernel')
        self.itakura_max_slope = config.get('DEFAULT', 'itakura_max_slope')
        self.sakoe_chiba_radius = config.get('DEFAULT', 'sakoe_chiba_radius')


def parse_args(isEntryFile):
    input_obj = Input()

    parser = argparse.ArgumentParser(usage=DTW_USAGE_MSG,
                             description=DTW_DESC_MSG,
                             formatter_class=argparse.RawDescriptionHelpFormatter,
                             add_help=False)

    # Control input arguments by terminal
    if isEntryFile:

        parser.add_argument('X',
                            type=argparse.FileType('r'),
                            help='POST file to be analyzed.')

        parser.add_argument('Y', nargs='?',
                            type=argparse.FileType('r'),
                            help='POST file to be analyzed.')

    else:
        parser.add_argument('-x', nargs='+', type=float, help="Temporal Serie 1")
        parser.add_argument('-y', nargs='+', type=float, help="Temporal Serie 2")

   
    parser.add_argument('-t', '--type_dtw', nargs='?', default=input_obj.MTS, type=str,
                        help="d: dependient or i: independient.")
    
    parser.add_argument("-d", "--local_dissimilarity", nargs='?', default=input_obj.local_dissimilarity, type=str,
                        help="Use a possible distance of scipy.spatial.distance, norm1, norm2, square_euclidean_distance or gower.")
   
    parser.add_argument("-ce", "--check_errors", nargs='?', default=input_obj.check_errors, type=str,
                        help="Control whether or not check for errors.")
    
    parser.add_argument("MTS", nargs='?', default=input_obj.MTS, type=bool,
                        help="Indicates whether the data are multivariate time series or not.")
    
    # Novelty
    parser.add_argument("-rf", "--regular_flag", nargs='?', default=input_obj.regular_flag, type=int,
                        help="Indicates the flag to irregular multivariate time series.")
   
    parser.add_argument("-vis", '--visualization', nargs='?', default=input_obj.visualization, type=bool,
                        help="Allows you to indicate whether to display the results or not. Only for API case.")
    
    parser.add_argument("-of", "--output_file", nargs='?', default=input_obj.output_file, type=bool,
                        help="Output by file instead of terminal.")
    
    parser.add_argument("-nf", "--name_file", nargs='?', default=input_obj.name_file, type=str,
                        help="Name file.")

    # In case of working with files containing N multivariate time series, we give the possibility 
    # to determine the number of threads and whether to transform the output into a kernel.
    parser.add_argument('-n', '--n_threads', nargs='?', default=input_obj.n_threads, type=int,
                        help="d: dependient or i: independient.")

    parser.add_argument("-k", "--DTW_to_kernel", nargs='?', default=input_obj.DTW_to_kernel, type=bool,
                        help="In case of DTW_to_kernel=True a transformation of the distance matrix to an exponential kernel with default sigma value equal to 1 is performed.")

    parser.add_argument("-s", "--sigma_kernel", nargs='?', default=input_obj.sigma_kernel, type=float,
                        help="Sigma value for kernel transformation.")

    # Novelties
    parser.add_argument("-imx", "--itakura_max_slope", nargs='?', default=input_obj.itakura_max_slope, type=str,
                    help="Maximum slope for the Itakura parallelogram. If None and select “itakura”, a maximum slope of 2.")

    parser.add_argument("-scr", "--sakoe_chiba_radius", nargs='?', default=input_obj.sakoe_chiba_radius, type=str,
                    help="Radius to be used for Sakoe-Chiba band. If None and select “sakoe_chiba”, a radius of 1 is used.")
    

    parser.add_argument('-h', '--help', action='help',
                    help=argparse.SUPPRESS)

    parser.add_argument('-v', '--version', action='version',
                    version=DTW_VERSION_MSG,
                    help=argparse.SUPPRESS)

    parser.add_argument('-g', '--debug', dest='debug',
                    action='store_true',
                    help=argparse.SUPPRESS)

    # Save de input arguments
    args = parser.parse_args()
    
    input_obj.type_dtw = args.type_dtw
    input_obj.local_dissimilarity = args.local_dissimilarity
    input_obj.check_errors = args.check_errors
    input_obj.MTS = args.MTS
    input_obj.regular_flag = args.regular_flag
    input_obj.visualization = args.visualization
    input_obj.output_file = args.output_file
    input_obj.name_file = args.name_file
    input_obj.n_threads = args.n_threads
    input_obj.DTW_to_kernel = args.DTW_to_kernel
    input_obj.sigma_kernel = args.sigma_kernel

    if args.itakura_max_slope == "None":
        input_obj.itakura_max_slope = None
    else:
        input_obj.itakura_max_slope = float(args.itakura_max_slope)

    if args.sakoe_chiba_radius == "None":
        input_obj.sakoe_chiba_radius = None
    else:
        input_obj.sakoe_chiba_radius = float(args.sakoe_chiba_radius)

    # If the distance introduced is not correct, the execution is terminated.
    if input_obj.check_errors:
        test_distance = possible_distances()
        if not input_obj.local_dissimilarity in test_distance:
            raise ValueError('Distance introduced not allowed or incorrect.')
         
    if not input_obj.local_dissimilarity in ["gower", "norm1", "norm2", "square_euclidean_distance"]:
       input_obj.local_dissimilarity = eval("distance." + input_obj.local_dissimilarity)


    return parser.parse_args(), input_obj
    
    
 
