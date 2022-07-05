import keras
from tqdm import tqdm
from scipy.io import loadmat
import pickle as pkl
import numpy as np
import random
from multiprocessing import Pool, cpu_count
import os
import math

class IQDataGenerator(keras.utils.Sequence):

    def __init__(self, ex_list, args):

        self.args = args
        self.ex_list = ex_list
        self.data_cache = {}
        

        # load all data to cache:
        print('Adding all files to cache')
        for ex in tqdm(self.ex_list):
            this_ex = loadmat(ex)['f_sig']
            data = np.zeros((this_ex.shape[1],2),dtype='float32')
            data[:,0] = np.real(this_ex[0,:])
            data[:,1] = np.imag(this_ex[0,:])
            self.__add_to_cache(ex, data)
        
        print 'len of ex_list and data_cache: ' 
        print len(self.ex_list), len(self.data_cache.keys())

    
     
    def __len__(self):
        # calculate total number of batches:
        with open (os.path.join(self.args.train_partition_path,'partition.pkl'),'rb') as handle:
            train_list = pkl.load(handle)['train']
        batch_cnt = math.ceil(1.0*len(train_list)/self.args.batch_size)
        if self.args.packet_aug:
            batch_cnt *= 0.05
        return batch_cnt



    def __add_to_cache(self, file, data):
       
        if self.args.normalize and not self.args.packet_aug:
            #data = (data - self.args.stats['mean']) / self.args.stats['std']
             # RMS normalization
             factor = np.sqrt(np.mean(np.sqrt(data[:,0]**2+data[:,1]**2)**2))
             data = data/factor 
        self.data_cache[file] = data
  
    

    def __getitem__(self, index):
        #Generate one batch of data 
        this_batch_ex_list = random.sample(self.data_cache.keys(), self.args.batch_size)

        X = np.zeros((self.args.batch_size, self.args.slice_size, 2), dtype='float32')
        y = np.zeros((self.args.batch_size, self.args.num_classes), dtype=int)

        cnt = 0
        for ex in this_batch_ex_list:
            
            this_ex = self.data_cache[ex]
            X[cnt,:,:] = this_ex

            cnt += 1

        # batch is ready, augment it if needed
        if self.args.cfo_aug:
           X,y =  cfo_augmentation(X, self.args.max_cfo, 'train')
        elif self.args.packet_aug:
            noise_power_list = list(map(lambda x : float(x.split('/')[-1].split('.ma')[0].split('_')[-1]), this_batch_ex_list))
            X, y = packet_augmentation(X, noise_power_list)

        return X, y

def cfo_augmentation(X, max_cfo, mode):
    X_augmented = np.zeros(X.shape)
    random_cfo = np.random.uniform(-max_cfo,max_cfo, X.shape[0])
    
    complex_X = X[:,:,0]+1j*X[:,:,1]

    t = 1.0*np.arange(160,320)/5000000       #5000000

    for i in range(X.shape[0]):

        #compensated_slice = complex_X[i,:]*np.exp(-2j*(math.pi)*cfo_batch[i]*t)
        compensated_slice = complex_X[i,:]

        #augmented_slice = compensated_slice*np.exp(2j*(math.pi)*random_cfo[i]*t)

        augmented_slice = np.multiply(compensated_slice , np.exp(2j*(np.pi)*random_cfo[i]*t))
        X_augmented[i,:,0] = np.real(augmented_slice)
        X_augmented[i,:,1] = np.imag(augmented_slice)
    
    #X_augmented[:,:,0] = np.real(np.multiply(complex_X, np.exp(2j*(np.pi)*random_cfo*t)))
    #X_augmented[:,:,1] = np.imag(np.multiply(complex_X, np.exp(2j*(np.pi)*random_cfo*t)))

    return X_augmented, random_cfo/max_cfo



def packet_augmentation(X, noise_power_list):

    #number of classes is 98 
    num_classes = 98
    y = np.zeros((X.shape[0],num_classes), dtype=int)
    # Flip a coin to decide to fill the void with high power noise or low power noise 
    coin = np.random.randint(0,2, size=X.shape[0])

    X_augmented = np.zeros(X.shape)
   
    # index in the lltf signal which is in index 159 of the input
    lltf_end_index = np.random.randint(0,159+1, size=X.shape[0])
  
    for i in range (X.shape[0]):
            
        this_signal_mean = np.mean(np.sqrt(X[i,:,0]**2+X[i,:,1]**2))
        this_signal_std = np.std(np.sqrt(X[i,:,0]**2+X[i,:,1]**2))

        #decide to fill the void with high power noise or low power
        if coin[i] == 0: # low noise
            desired_noise_scale = np.sqrt(noise_power_list[i])
        else:
            desired_noise_scale = this_signal_std

        useful_LLTF = X[i,0:lltf_end_index[i]+1,:]
        bed_of_noise = np.random.normal(loc=this_signal_mean, scale=desired_noise_scale, size=(X.shape[1],2))
        this_start = 160 - lltf_end_index[i] - 1
        bed_of_noise[this_start:this_start+lltf_end_index[i]+1,:] = useful_LLTF
        X_augmented[i,:,:] = bed_of_noise
        
        if lltf_end_index[i] > 62:                 # the coin wants us to create a valid LLTF
            y[i,this_start] = 1
        else:                           # the coin wants us to create an invalid LLTF, the end should be <= 62
           y[i,97] = 1

    """#if we are in training
    random_index = np.random.randint(0,160,size=4)
    # we force this slice to have all noise
    X_augmented[random_index[0],:,:]= np.random.normal(loc=this_signal_mean, scale=np.sqrt(noise_power_list[random_index[0]]), size=(X.shape[1],2))
    y[random_index[0],97] = 1
    
    # force this slice to be all L-LTF
    X_augmented[random_index[1],:,:]= X[random_index[1],:,:] 
    y[random_index[1],0] = 1"""
        
        
    # right before sending out
    # do an rms normalization
    for i in range (X_augmented.shape[0]):
        factor = np.sqrt(np.mean(np.sqrt(X_augmented[i,:,0]**2+X_augmented[i,:,1]**2)**2))
        X_augmented[i,:,:] = X_augmented[i,:,:]/factor


    return X_augmented, y 


if __name__ == "__main__":
    base_path = '/home/PRONTO/datasets/Oracle-NotCompensated-LLTF/' 
    class Employee:
        pass
    args = Employee()

    with open (base_path + 'partition.pkl','rb') as handle:
        partition = pkl.load(handle)
    ex_list = partition['train']


    args.normalize = True
    args.batch_size = 256
    args.slice_size = 160 
    args. num_classes = 98 
    args.cfo_aug = False
    args.cfo_estimation = False
    args.packet_aug = True

    DG = IQDataGenerator(ex_list, args)

    print DG.__getitem__(0)
