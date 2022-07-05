from tqdm import tqdm
import os
import glob
from scipy.io import loadmat,savemat
import random
import pickle
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import base64
import struct

def create_dataset(base_address, dataset_address):
    """ for each class it gets the paths to all the mat files associated with that class and creates a list of the paths
    then we shuffle the list and pick 70% for training, 10% for validation, 20% for test"""
    if not os.path.isdir(dataset_address):
        os.mkdir(dataset_address)

    print "Creating train/val/test partitions:"
    
    all_mat_files = glob.glob(base_address+'*')
    
    random.shuffle(all_mat_files)
           
    """train_list = all_mat_files[:int(0.7*len(all_mat_files))]
    val_list = all_mat_files[int(0.7*len(all_mat_files)):int(0.8*len(all_mat_files))]
    test_list = all_mat_files[int(0.8*len(all_mat_files)):]"""

    train_list, val_list = [],[]
    test_list = all_mat_files


    # partition 
    print "Creating partition pickle file:"
    partition = {}
    partition['train']= train_list
    partition['test']= test_list 	
    partition['val']= val_list
    print "length of partitions is:"
    print len(train_list),len(val_list),len(test_list)
    with open (dataset_address+'partition.pkl','wb') as handle:
        pickle.dump(partition,handle)


    print "Dataset successfully generated!"    
    print dataset_address

if __name__ == "__main__": 
    
    base_address = '/home/PRONTO/mat_files/Oracle-Compensated-With-Phase-Noise-Large/'
    dataset_address = '/home/PRONTO/datasets/Oracle-Compensated-With-Phase-Noise-Large/'
    
    #base_address = '/home/PRONTO/mat_files/Arena-NotCompensated-LLTF/'
    #dataset_address = '/home/PRONTO/datasets/Arena-NotCompensated-LLTF/'
    
    create_dataset(base_address, dataset_address)
