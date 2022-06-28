import csv
import sys
import pandas as pd
import os.path
import numpy as np

# Created functions
from .error_control import possible_distances
from .utils import *
from .dtw_functions import dtw, dtw_tensor_3d



def input_File(input_obj):

    args, input_obj = parse_args(True)
    data = read_data(args.X)

    if (data.shape[0] == 2) and (data.shape[0] % 2 == 0):
        input_obj.MTS = False
        input_obj.x = data.iloc[0,:].values
        input_obj.y = data.iloc[1,:].values

    elif (data.shape[0] > 3) and (data.shape[0] % 2 == 0):
        input_obj.MTS = True
        finalData = []
        index = data.shape[0]/2
        for i in range(2):
            finalData.append(data.loc[(data.shape[0]/2)*i:index-1, :].values)
            index+=int(data.shape[0] / 2)

        input_obj.x = finalData[0]
        input_obj.y = finalData[1]

    return input_obj


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
            input_obj = input_File(input_obj)
            dtw_distance = dtw(input_obj.x, input_obj.y, input_obj.type_dtw, input_obj.distance,
                               input_obj.MTS, input_obj.visualization, input_obj.check_errors)

        # input 3D file. We include the possibility to parallelise.
        elif sys.argv[1].endswith('.npy'):

            args, input_obj = parse_args(True)
            if len(sys.argv) > 2 and sys.argv[2].endswith('.npy'):
                X, Y = read_npy(args.X), read_npy(args.Y)
            else:
                X = read_npy(args.X)
                Y = X

            dtw_distance = dtw_tensor_3d(X, Y, input_obj)
            
        if input_obj.output_file:
            print("output to file.")
            pd.DataFrame(np.array([dtw_distance])).to_csv("output.csv", index=False)
        else:
            sys.stdout.write(str(dtw_distance))   
            
    # Input type 2: input by terminal
    elif sys.argv[1] == "-x":        
        args, input_obj = parse_args(False)

        input_obj.x = args.x
        input_obj.y = args.y
       
        dtw_distance = dtw(input_obj.x, input_obj.y, input_obj.type_dtw, input_obj.distance, input_obj.MTS,
                           input_obj.visualization, input_obj.check_errors)
        
        sys.stdout.write(str(dtw_distance))
        
    else:
        s = "\nError in input arguments.\n"
        sys.stderr.write(s)
        sys.exit(0)


if __name__ == "__main__":

   try:
      main()
   except KeyboardInterrupt:
      s = "\n\nReceived Ctrl-C or other break signal. Exiting.\n"
      sys.stderr.write(s)
      sys.exit(0)



