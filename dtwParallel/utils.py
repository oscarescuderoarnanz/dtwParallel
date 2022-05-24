import pandas as pd
import numpy as np
import argparse

# Function to convert string to boolean
def str_to_bool(a):
    if a == "True":
        return True
        
    return False
    
# Function to convert string to float
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
    parser = argparse.ArgumentParser(description='Read POST run outputs.')
    parser.add_argument('file',
                        type=argparse.FileType('r'),
                        help='POST file to be analyzed.')
    return parser.parse_args()
    
    
 
