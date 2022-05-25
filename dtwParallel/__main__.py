import argparse
import csv
import sys
import pandas as pd
from scipy.spatial import distance
import os.path
from dtw_functions import dtw, dtw_tensor_3d
import numpy as np
import configparser
from configuration import create_file_ini
from error_control import possible_distances
from utils import *



DTW_USAGE_MSG = \
    """%(prog)s [<args>] | --help | --version | --list"""

DTW_DESC_MSG = \
    """
    Args:
	   -x   Time series 1
	   -y   Time series 2
	   -ec  Error control
	   -d   Type of distance
	   -t   Calculation type DTW
    
    Optional arguments:
      -h, --help            show this help message and exit
      -v, --version         show version
      -l, --list            show available backends
"""
    
DTW_VERSION_MSG = \
    """%(prog)s 0.0.5"""


class Input:
    def __init__(self):
        Input.execute_configuration(self)
        config = configparser.ConfigParser()
        config.read('./configuration.ini')
        self.errors_control = config.getboolean('DEFAULT', 'errors_control')
        self.type_dtw = config.get('DEFAULT', 'type_dtw')
        self.MTS = config.getboolean('DEFAULT', 'MTS')
        self.n_threads = config.getint('DEFAULT', 'n_threads')
        
        # If the distance introduced is not correct, the execution is terminated.
        if self.errors_control:
            test_distance = possible_distances()
            if not config.get('DEFAULT', 'distance') in test_distance:
                raise ValueError('Distance introduced not allowed or incorrect.')
             
        #self.distance = eval("distance." + config.get('DEFAULT', 'distance'))
        self.distance = config.get('DEFAULT', 'distance')
        self.visualization = config.getboolean('DEFAULT', 'visualization')
        self.output_file = config.getboolean('DEFAULT', 'output_file')
        self.DTW_to_kernel = config.getboolean('DEFAULT', 'DTW_to_kernel')
        self.sigma = config.getint('DEFAULT', 'sigma')

    def execute_configuration(self):
        create_file_ini()


def input_File(input_obj):

    args = parse_args()
    data = read_data(args.file)
    
    if (data.shape[0] == 2) and (data.shape[0] % 2 == 0):
        input_obj.MTS = False
        x = string_to_float(data.iloc[0, :].values[0].split(';'))
        y = string_to_float(data.iloc[1, :].values[0].split(';'))

    elif (data.shape[0] > 3) and (data.shape[0] % 2 == 0):
        input_obj.MTS = True
        x = []
        for i in range(int(data.shape[0] / 2)):
            x.append(string_to_float(data.iloc[0:int(data.shape[0] / 2), :].values[i][0].split(';')))
        y = []
        for i in range(int(data.shape[0] / 2)):
            y.append(string_to_float(data.iloc[int(data.shape[0] / 2):int(data.shape[0]), :].values[i][0].split(';')))

    input_obj.x = x
    input_obj.y = y
    input_obj.distance = eval("distance."+input_obj.distance)

    return dtw(input_obj.x, input_obj.y, input_obj.type_dtw, input_obj.distance,
     input_obj.MTS, input_obj.visualization, input_obj.errors_control), input_obj.output_file


   
def main():
	
	# Generate an object with the deafult parameters
    input_obj = Input()
	
    # Input type 1: input by csv file
    if len(sys.argv) == 2 and os.path.exists(sys.argv[1]):
        # input 2D file
        if sys.argv[1].endswith('.csv'):
            dtw_distance, output_file = input_File(input_obj)
        # input 3D file. We include the possibility to parallelise.
        elif sys.argv[1].endswith('.npy'):
            args = parse_args()
            X = read_npy(args.file)
            input_obj.distance = eval("distance."+input_obj.distance)
            input_obj.DTW_to_kernel = str_to_bool(input_obj.DTW_to_kernel)
            dtw_distance = dtw_tensor_3d(X, X, input_obj)
        else:
            raise ValueError('Error in load file.')
            
        if input_obj.output_file:
            print("output to file")
            pd.DataFrame(np.array([dtw_distance])).to_csv("output.csv", index=False)
        else:
            return dtw_distance
            
    # Input type 2: input by terminal
    else:

        # Control input arguments by terminal
        parser = argparse.ArgumentParser(usage=DTW_USAGE_MSG,
                                     description=DTW_DESC_MSG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     add_help=False)

        parser.add_argument('-h', '--help', action='help',
                        help=argparse.SUPPRESS)
        parser.add_argument('-v', '--version', action='version',
                        version=DTW_VERSION_MSG,
                        help=argparse.SUPPRESS)
        parser.add_argument('-g', '--debug', dest='debug',
                        action='store_true',
                        help=argparse.SUPPRESS)

                        
        parser.add_argument('-x', nargs='+', type=int, help="Temporal Serie 1")
        parser.add_argument('-y', nargs='+', type=int, help="Temporal Serie 2")
        parser.add_argument("-ec", "--errors_control", nargs='?', default=input_obj.errors_control, type=str, help="Control whether or not check for errors.")
        parser.add_argument("-d", "--distance", nargs='?', default=input_obj.distance, type=str, help="Use a possible distance of scipy.spatial.distance.")
        parser.add_argument('-t', '--type_dtw', nargs='?', default=input_obj.MTS, type=str, help="d: dependient or i: independient.")
        parser.add_argument("MTS", nargs='?', default=input_obj.type_dtw, type=bool, help="Indicates whether the data are multivariate time series or not.")
        parser.add_argument("visualization", nargs='?', default=input_obj.visualization, type=bool, help="Allows you to indicate whether to display the results or not. Only for the one-dimensional case.")
        
        # Save de input arguments
        args = parser.parse_args()
        input_obj.x = args.x
        input_obj.y = args.y
        input_obj.errors_control = args.errors_control
        input_obj.type_dtw = args.type_dtw
        input_obj.distance = args.distance

        
        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(1)
        
        # Convert boolean parameters introduced by terminal
        input_obj.MTS = str_to_bool(args.MTS)
        input_obj.visualization = str_to_bool(args.visualization)
    
      
		# If the distance introduced is not correct, the execution is terminated.
        if input_obj.errors_control:
            test_distance = possible_distances()
            if not input_obj.distance in test_distance:
                raise ValueError('Distance introduced not allowed or incorrect.')
             
        input_obj.distance = eval("distance." + input_obj.distance)
		
        return dtw(input_obj.x, input_obj.y, input_obj.type_dtw,
         input_obj.distance, input_obj.MTS, input_obj.visualization,
         input_obj.errors_control)


if __name__ == "__main__":
    main()



