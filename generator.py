import argparse
import os
import pickle
import torch
from run import load_model, save_image

import config
import network




parser = argparse.ArgumentParser(description='Generate frame from landmark')
parser.add_argument("--continue_id", required=True, help='continue_id of the generator model')
parser.add_argument("--e_hat", required=True, help='path to e_hat tensor')
parser.add_argument("--landmark", required=True, help = 'path to landmark tensor')
args = parser.parse_args()

continue_id = args.continue_id
y_t = torch.load(args.landmark)
e_hat =torch.load(args.e_hat)

G = network.Generator(0)
G = load_model(G, continue_id)

x_hat = G(y_t, e_hat)

save_image(os.path.join(config.GENERATED_DIR, f'Generated_x_hat.png'), x_hat[0])
