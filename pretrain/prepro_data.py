import os
import numpy as np
import pickle
from collections import OrderedDict

seq_home = '../dataset/'
seqlist_path = 'data/vot-otb.txt'
output_path = 'data/vot-otb.pkl'

with open(seqlist_path, 'r') as fp:
    seq_list = fp.read().splitlines()

data = {}
for i, seq in enumerate(seq_list):
    print(seq)
    img_list = sorted([p for p in os.listdir(seq_home+seq) if os.path.splitext(p)[1] == '.jpg'])
    if seq == 'vot2014/ball':
        img_list.pop()  # delete the last element, because there are 603 images but gt has only 602

    gt = np.loadtxt(seq_home+seq+'/groundtruth.txt',delimiter=',')
    print(len(img_list))
    print(len(gt))
    assert len(img_list) == len(gt), "Lengths do not match!!"

    if gt.shape[1] == 8:  # shape[0] is rows, shape[1] is columns
        x_min = np.min(gt[:,[0,2,4,6]],axis=1)[:,None]  # the min return a row, but we need a column, so add [:,None]
        y_min = np.min(gt[:,[1,3,5,7]],axis=1)[:,None]  # None can add a new axis
        x_max = np.max(gt[:,[0,2,4,6]],axis=1)[:,None]
        y_max = np.max(gt[:,[1,3,5,7]],axis=1)[:,None]
        gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

    data[seq] = {'images':img_list, 'gt':gt}  # the seqname is stored in the key of the dictionary, the seq!
    # insert randomly because it's a dictionary!

with open(output_path, 'wb') as fp:
    pickle.dump(data, fp, -1)  # If protocol is specified as a negative value or HIGHEST_PROTOCOL,
    #  the highest protocol version will be used.
