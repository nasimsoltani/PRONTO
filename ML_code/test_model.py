import os
import pickle as pkl
from scipy.io import loadmat
import collections
import numpy as np
from tqdm import tqdm
import data_generator as DG

def test_model(args, model):

    # initialize:

    correct_slice_cntr = 0
    total_slice_cntr = 0
    correct_ex_cntr = 0
    total_ex_cntr = 0

    Preds = {}
    preds_slice = {}
    preds_ex = {}

    # load test set
    with open(os.path.join(args.test_partition_path,'partition.pkl'),'rb') as handle:
        test_list = pkl.load(handle)['test']
        #test_list = pkl.load(handle)['val']
    print 'length of test list is: ' +str(len(test_list))   
    
    start_index = 0
    squared_error = 0
    true_labels = {}
    results = {}
    
    pbar = tqdm(total = len(test_list))
    while start_index < len(test_list)-1:
        left = len(test_list)-1-start_index
        jump = min(args.batch_size,left)
        
        print('this many examples are left: '+str(left))

        X = np.zeros((jump,args.slice_size,2),dtype='float32')
        batch_index = 0
        ex_list = []
        for ex in test_list[start_index:start_index+jump]:
            # loading mat file
            this_ex = loadmat(ex)['f_sig']
            X_real = np.real(this_ex[0,:])
            X_imag = np.imag(this_ex[0,:])
            if args.normalize and not args.packet_aug:
                #X = (X - args.stats['mean']) / args.stats['std']
                # RMS normalization
                factor = np.sqrt(np.mean(np.sqrt(X_real**2+X_imag**2)**2))
                X_real = X_real/factor 
                X_imag = X_imag/factor 
 
            X[batch_index,:,0] = X_real
            X[batch_index,:,1] = X_imag
            batch_index += 1
            ex_list.append(ex)

        start_index += args.batch_size
    
       
        if args.cfo_estimation and args.cfo_aug:
            X,y = DG.cfo_augmentation(X, args.max_cfo, 'test')
        elif args.packet_aug:
            noise_power_list = list(map(lambda x : float(x.split('/')[-1].split('.ma' )[0].split('_')[-1]), ex_list))
            X,y = DG.packet_augmentation(X, noise_power_list)
        
        preds = model.predict(X, batch_size=args.batch_size)

        for index in range(preds.shape[0]):
            ex = ex_list[index]

            if args.cfo_estimation:
                squared_error += (y[index]-preds[index][0])**2
                results[ex] = preds[index][0]
                true_labels[ex] = y[index]

            else:
                # calculate example accuracy:
                true_index = np.argmax(y[index])
                example_pred_index = np.argmax(preds[index])
                if example_pred_index == true_index:
                    correct_ex_cntr += 1
                total_ex_cntr +=1
                results[ex] = preds[index]
                true_labels[ex] = y[index]
    pbar.close()

    Preds['true_labels'] = true_labels
    Preds['results'] = results
 
    if args.cfo_estimation:
        metric = squared_error / len(test_list) 
    else:
        metric = 1.0*correct_ex_cntr/total_ex_cntr
        
    # save preds:
    with open(os.path.join(args.save_path,'preds.pkl'),'wb') as handle:
        pkl.dump(Preds, handle)
    
    return metric

