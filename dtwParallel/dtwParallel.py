import csv
import sys
import pandas as pd
import os.path
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Created functions
from error_control import possible_distances
from utils import *
from dtw_functions import dtw, dtw_tensor_3d



def input_File():
    """
    For terminal. Controls the file entered. 
    In case of having 2 rows, we calculate DTW between 2 MTS. 
    In case of having N rows (pairs), we associate N/2 to each MTS, calculating the DTW distance between 2 MTS.
    In case of having N rows not contemplated
    """
    args, input_obj = parse_args(True)
    data = read_data(args.X)

    if (data.shape[0] == 2) and (data.shape[0] % 2 == 0):
        input_obj.MTS = False
        x = data.iloc[0,:].values
        y = data.iloc[1,:].values
        input_obj.x = [[value] for value in x]
        input_obj.y = [[value] for value in y]

    elif (data.shape[0] > 3) and (data.shape[0] % 2 == 0):
        input_obj.MTS = True
        final_data = []
        index = data.shape[0]/2
        for i in range(2):
            final_data.append(data.loc[(data.shape[0]/2)*i:index-1, :].values)
            index+=int(data.shape[0] / 2)

        input_obj.x = final_data[0]
        input_obj.y = final_data[1]

    return input_obj


def control_output(input_obj, dtw_distance):

    if input_obj.output_file:
        sys.stdout.write("Output to "  + input_obj.name_file + ".csv")
        pd.DataFrame(np.array([dtw_distance])).to_csv(input_obj.name_file + ".csv", float_format='%g', index=False)
    else:
        sys.stdout.write(str(dtw_distance)+"\n")


def main():
    # If you only make use of the library you will get an error.
    if len(sys.argv) == 1:
        s = "\nIt needs input arguments.\n"
        sys.stderr.write(s)
        sys.exit(0)
	
    # Generate an object with the deafult parameters
    input_obj = Input()
	
    # Input type 1: input by files
    if os.path.exists(sys.argv[1]):
        # input 2D file
        if sys.argv[1].endswith('.csv'):
            input_obj = input_File()
            
            dtw_distance = dtw(input_obj.x, input_obj.y, type_dtw=input_obj.type_dtw, constrained_path_search=input_obj.constrained_path_search, local_dissimilarity=input_obj.local_dissimilarity, MTS=input_obj.MTS, get_visualization=input_obj.visualization, check_errors=input_obj.check_errors, term_exec=True)

        # input 3D file. We include the possibility to parallelise.
        elif sys.argv[1].endswith('.npy'):

            args, input_obj = parse_args(True)
            input_obj.MTS = True
            if len(sys.argv) > 2 and sys.argv[2].endswith('.npy'):
                X, Y = read_npy(args.X), read_npy(args.Y)
            else:
                X = read_npy(args.X)
                Y = X

            dtw_distance = dtw_tensor_3d(X, Y, input_obj)
            
        control_output(input_obj, dtw_distance)
            
    # Input type 2: input by terminal
    elif sys.argv[1] == "-x":        
        args, input_obj = parse_args(False)
        
        if args.y is None:
            sys.stderr.write("You need introduce a vector -y")
            sys.exit(0)
      
        input_obj.x = [[value] for value in args.x]
        input_obj.y = [[value] for value in args.y]

        dtw_distance = dtw(input_obj.x, input_obj.y, type_dtw=input_obj.type_dtw, constrained_path_search=input_obj.constrained_path_search, local_dissimilarity=input_obj.local_dissimilarity, MTS=input_obj.MTS, get_visualization=input_obj.visualization, check_errors=input_obj.check_errors, term_exec=True)
        
        control_output(input_obj, dtw_distance)
        
    else:
        _, _ = parse_args(False)
        sys.exit(0)


if __name__ == "__main__":

   try:
      main()
   except KeyboardInterrupt:
      s = "\n\nReceived Ctrl-C or other break signal. Exiting.\n"
      sys.stderr.write(s)
      sys.exit(0)



