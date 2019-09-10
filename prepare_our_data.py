'''
input: two folders mouth_folder and crd_folder
ouput: .vid file
'''

import os
import argparse
import cv2
import numpy as np
import pickle as pkl
import random


parser = argparse.ArgumentParser(description='Prepare .vid file')
parser.add_argument("--m", required=True, help='mouth_folder')
parser.add_argument("--c", required=True, help='crd_folder')
parser.add_argument("--o", required=True, help = 'path where to put .vid')
args = parser.parse_args()

data = []

for mouth in os.listdir(args.m):
    crdname = mouth[:-3] + 'npy'
    crd = np.load(os.path.join(args.c,crdname))

    img = cv2.imread(os.path.join(args.m,mouth))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(img,(250,350))

    # cv2.imwrite('frame.png',frame)


    data.append({
        'frame': frame,
        'landmarks': crd,
    })

print('len data: ', len(data))

for i in range(16):
    filename = 's2_{}.vid'.format(i)
    idx = random.sample(range(0,len(data)),30)
    small_data = [data[k] for k in idx]
    pkl.dump(small_data, open(os.path.join(args.o, filename), 'wb'))
    print('Saved file: {}'.format(filename))
