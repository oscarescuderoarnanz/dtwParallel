import argparse
import csv
import sys
import pandas as pd
from scipy.spatial import distance
import os.path
from dtw_functions import dtw, dtw_tensor_3d
from error_control import validate
import ast
import numpy as np
import configparser



DTW_USAGE_MSG = \
    """%(prog)s [<args>] | --help | --version | --list"""

DTW_DESC_MSG = \
    """
    Args:
	   -x   Time series 1
	   -y   Time series 2
	   -d   Type of distance
	   -t   Calculation type DTW 
    
    Optional arguments:
      -h, --help            show this help message and exit
      -v, --version         show version
      -l, --list            show available backends
"""
    
DTW_VERSION_MSG = \
    """%(prog)s 0.0.1"""


class Input:
    def __init__(self):
        #config = configparser.ConfigParser()
        #config.read('./dtwParallel/configuration.ini')
        #print(config['DEFAULT'].items())
        #print(config['DEFAULT'].values())
        #print(config.getboolean('DEFAULT', 'errors_control'))
        #print(config.values())
        self.conf_path = './dtwParallel/conf.txt'
        constant_parameters = Input.read_configuration(self)
        self.errors_control = bool(constant_parameters[0])
        self.type_dtw = bool(constant_parameters[1])
        self.MTS = bool(constant_parameters[2])
        self.verbose = int(constant_parameters[3])
        self.n_threads = int(constant_parameters[4])
        self.distance = constant_parameters[5]
        #self.distance = distance.euclidean
        self.visualization = ast.literal_eval(constant_parameters[6])
        self.output_file = ast.literal_eval(constant_parameters[7])

    def read_configuration(self):
        with open(self.conf_path) as f:
            lines = f.readlines()
            f.close()
        split_lines = []
        for i in range(len(lines)):
            split_lines.append(lines[i].split('=')[1].replace('\n', ""))
        return split_lines



def string_to_float(data):
    arr_float = []

    for i in range(len(data)):
        arr_float.append(float(data[i]))

    return arr_float


def read_data(fname):
    return pd.read_csv(fname, header=None)

def read_npy(fname):
    return np.load(fname)


def parse_args():
    print(pd.__file__)
    parser = argparse.ArgumentParser(description='Read POST run outputs.')
    parser.add_argument('file',
                        type=argparse.FileType('r'),
                        help='POST file to be analyzed.')
    return parser.parse_args()


def input_File():

    args = parse_args()
    data = read_data(args.file)
    input_obj = Input()
    if (data.shape[0] == 2) and (data.shape[0] % 2 == 0):
        input_obj.MTS = False
        print("Caso univariante")
        x = string_to_float(data.iloc[0, :].values[0].split(';'))
        y = string_to_float(data.iloc[1, :].values[0].split(';'))

    elif (data.shape[0] > 3) and (data.shape[0] % 2 == 0):
        input_obj.MTS = True
        x = []
        for i in range(int(data.shape[0] / 2)):
            x.append(string_to_float(data.iloc[0:int(data.shape[0] / 2), :].values[i][0].split(';')))
        y = []
        for i in range(int(data.shape[0] / 2)):
            y.append(
                string_to_float(data.iloc[int(data.shape[0] / 2):int(data.shape[0]), :].values[i][0].split(';')))

    input_obj.x = x
    input_obj.y = y

    if input_obj.errors_control:
        validate(input_obj)

    return dtw(input_obj.x, input_obj.y, input_obj.type_dtw, input_obj.distance, input_obj.MTS, input_obj.visualization), input_obj.output_file


def main():
	
    # Input type 1: input by csv file
    #print(sys.argv)
    #if len(sys.argv) == 2 and not sys.argv[1] in ["-h", "-v", "-g"]:
    if len(sys.argv) == 2 and os.path.exists(sys.argv[1]):
        # input 2D file
        if sys.argv[1].endswith('.csv'):
            dtw_distance, output_file = input_File()
        # input 3D file
        elif sys.argv[1].endswith('.npy'):
            args = parse_args()
            X = read_npy(args.file)
            type_dtw = "d"
            dist = distance.euclidean
            dtw_distance, output_file = dtw_tensor_3d(X, X, type_dtw, dist)
        else:
            raise ValueError('Error in load file.')
            
        if output_file:
            print("output to file")
            pd.DataFrame(np.array([dtw_distance])).to_csv("output.csv", index=False)
        else:
            print(dtw_distance)
            
    # Input type 2: input by terminal
    else:
        
        input_obj = Input()
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

                        
        # implementar para prueba sencilla
        parser.add_argument('-x', nargs='+', type=int, help="Temporal Serie 1")
        parser.add_argument('-y', nargs='+', type=int, help="Temporal Serie 2")
        parser.add_argument("-d", "--distance", type=str, help="Use a possible distance of scipy.spatial.distance")
        parser.add_argument('-t', '--type_dtw', type=str, help="d: dependient or i: independient")
        parser.add_argument("MTS", choices=('True','False'))
        parser.add_argument("visualization", choices=('True','False'))
        
        args = parser.parse_args()
        print(args)
        input_obj.x = args.x
        input_obj.y = args.y
        input_obj.type_dtw = args.type_dtw
        input_obj.distance = args.distance
            
        
        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(1)
        
        if args.MTS == "True":
            input_obj.MTS = True
        else:
            input_obj.MTS = False
    
        if args.visualization == "True":
            input_obj.visualization = True
        else:
            input_obj.visualization = False
            
        possible_distance = []
        for i in range(len(dir(distance))):
            if(len(dir(distance)[i].split("_")) == 1) and not any(c.isupper() for c in dir(distance)[i]):
                possible_distance.append(dir(distance)[i])
				
        if not input_obj.distance in possible_distance:
             sys.exit(1)
    
        print(dtw(input_obj.x, input_obj.y, input_obj.type_dtw, distance.euclidean, input_obj.MTS, input_obj.visualization))

if __name__ == "__main__":
    main()
