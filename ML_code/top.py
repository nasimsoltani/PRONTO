import argparse
from keras.callbacks import TensorBoard
from time import time
from keras.utils import plot_model
import os
import pickle
from keras.optimizers import Adam

from create_model import *
from train_model import *
from test_model import *

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train and validation pipeline',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exp_name', default='exp1', type=str, help='experiment name')
    parser.add_argument('--save_path', default='', type=str, help='The path where you want the results to be saved')
    parser.add_argument('--train_partition_path', default='', type=str, help='Specify the train path')
    parser.add_argument('--test_partition_path', default='', type=str, help='Specify the test path')
    parser.add_argument('--stats_path', default='/scratch/RFMLS/dataset100/dataset_with_val_9000train/', type=str, help='Specify the stats path')
    parser.add_argument('--model_flag', default='baseline', type=str, help='Specify which model to use')
    parser.add_argument('--packet_aug', default=False, type=str2bool, help='Specify whether to use packet start augmentation or not')
    parser.add_argument('--cfo_aug', default=False, type=str2bool, help='Specify whether to use cfo augmentation or not')
    parser.add_argument('--cfo_estimation', default=False, type=str2bool, help='Specify whether to use cfo augmentation or not')
    parser.add_argument('--max_cfo', default=39000, type=int, help='The highest CFO that we can have.')

    parser.add_argument('--slice_size', default=1024, type=int, help='Specify the slice size')
    parser.add_argument('--batch_size', default=32, type=int, help='Specify the batch size')
    parser.add_argument('--num_classes', default=100, type=int, help='Specify the number of total devices')
    parser.add_argument('--normalize', default='True', type=str2bool, help='Specify if you want to normalize the data using mean and std in stats files (if stats does not have this info, it is ignored)')
    parser.add_argument('--epochs', default=10, type=int, help='')
    parser.add_argument('--id_gpu', default=0, type=int, help='If --multigpu=False, this arguments specify which gpu to use.')
    parser.add_argument('--early_stopping', default=False, type=str2bool, help='Specify if you want to use early stopping')
    parser.add_argument('--patience', default=1, type=int, help='patience')
    parser.add_argument('--train', default=False, type=str2bool, help='Specify doing training or not')
    parser.add_argument('--test', default=False, type=str2bool, help='Specify doing Test or not')
    parser.add_argument('--contin', default=False, type=str2bool, help='If you want to load a pre-trained model')
    parser.add_argument('--hdf5_path', default='', type=str, help='weight path')
    parser.add_argument('--json_path', default='', type=str, help='model path')

    # initialize:
    args = parser.parse_args()
    if args.id_gpu >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # The GPU id to use
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)
    args.save_path = os.path.join(args.save_path , args.exp_name)

    #print args
    
    # create the model
    model = create_model(args)
    model.summary()

    if args.cfo_estimation:
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mse'])
    else:
        #model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), metrics=['accuracy'])

    # train the model if train is true
    if args.train:
        print('***** Training the model *****')
        model = train_model(args, model)

    # test the model if test is true
    if args.test:
        print('***** Testing the model *****')
        metric = test_model(args, model)

        print('accuracy or mean squared error is:')
        print(metric)
