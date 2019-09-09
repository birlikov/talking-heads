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
parser.add_argument("--mouth_folder", required=True, help='mouth_folder')
parser.add_argument("--crd_folder", required=True, help='crd_folder')
parser.add_argument("--outputdir", required=True, help = 'path where to put .vid')
args = parser.parse_args()

n_frames = len(os.listdir(args.mouth_folder))
h = 150
w = 250

data = []

for mouth in os.listdir(args.mouth_folder):
    crdname = mouth[:-3] + 'npy'
    crd = np.load(os.path.join(args.crd_folder,crdname))

    img = cv2.imread(os.path.join(args.mouth_folder,mouth))
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    data.append({
        'frame': frame,
        'landmarks': crd,
    })

print('len data: ', len(data))

for i in range(10):
    filename = 's2_{}.vid'.format(i)
    idx = random.sample(range(0,len(data)),20)
    small_data = [data[k] for k in idx]
    pkl.dump(small_data, open(os.path.join(args.outputdir, filename), 'wb'))
    print('Saved file: {}'.format(filename))
