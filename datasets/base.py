"""
Base IO code for datasets
"""

# author ShuoshuoFan
# Email  shuoshuofan@gmail.com

import os
import csv
import sys
import shutil
from os  import environ
from os.path import dirname
from os.path import join
from os.path import exists
from os.path import expanduser
from os.path import  isdir
from os.path import splitext
from os import listdir
from os import makedirs

import numpy as np
from pandas  import read_csv

def load_data(module_path,data_file_name):

   # module_path='/home/shuoshuo/git/ML_Python/datasets/'
   # data_file_name='iris.csv'
    with open(join(module_path,data_file_name)) as csv_file:
        data_file=csv.reader(csv_file)
        temp =next(data_file)
        n_samples=int(temp[0])
        n_features=int(temp[1])
        target_names=np.array(temp[2:])
        data=np.empty((n_samples,n_features))
        target=np.empty((n_samples),dtype=np.int)
        data_internal=[row for row in data_file]
        for i ,ir in enumerate(data_internal):
            data[i]=np.asarray(ir[:-1],dtype=np.float64)
            target[i]=np.asarray(ir[-1],dtype=np.int)

    return  data_internal,target
"""
def load_bankdata(module_path,data_file_name):

    with open(join(module_path,data_file_name)) as csv_file:
        data_file=csv.reader(csv_file)
        next(data_file)
        n_samples=0
        for row in data_file:
            n_samples+=1

        n_features=64
        data=np.empty((n_samples,n_features))
        target=np.empty((n_samples))
        row_number=0
        for row in data_file:
            data[row_number]=np.asarray(row[:-1],dtype=np.float64)
            target[row_number]=np.asarray(row[-1])
            row_number+=1
        return  data,target
"""

def load_bankdata(module_path,data_file_name):

    bank_data=read_csv(join(module_path,data_file_name))
    tmp=bank_data.as_matrix()

    data= tmp[:,:-1]
    target=tmp[:,-1]
    return data,target

def load_iris():

    module_path='/home/shuoshuo/git/ML_Python/datasets/data'

    data_file_name='iris.csv'
    data,target =load_data(module_path,data_file_name)
    return data,target

def load_1year():

    module_path='/home/shuoshuo/git/ML_Python/datasets/data/bank'
    data_file_name='1year.arff.csv'
    data,target =load_bankdata(module_path,data_file_name)
    return data,target

def load_2year():

    module_path='/home/shuoshuo/git/ML_Python/datasets/data/bank'
    data_file_name='2year.arff.csv'
    data,target =load_bankdata(module_path,data_file_name)
    return data,target

def load_3year():

    module_path='/home/shuoshuo/git/ML_Python/datasets/data/bank'
    data_file_name='3year.arff.csv'
    data,target =load_bankdata(module_path,data_file_name)
    return data,target

def load_4year():

    module_path='/home/shuoshuo/git/ML_Python/datasets/data/bank'
    data_file_name='4year.arff.csv'
    data,target =load_bankdata(module_path,data_file_name)
    return data,target

def load_5year():

    module_path='/home/shuoshuo/git/ML_Python/datasets/data/bank'
    data_file_name='5year.arff.csv'
    data,target =load_bankdata(module_path,data_file_name)
    return data,target
