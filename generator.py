import argparse
import os
import pickle
from run import load_model, save_image

import config
import network



parser = argparse.ArgumentParser(description='Generate frame from landmark')
parser.add_argument("--continue_id", required=True, help='continue_id of the generator model')
parser.add_argument("--e_hat", required=True, help='path to e_hat tensor')
parser.add_argument("--landmark", required=True, help = 'path to landmark tensor')
args = parser.parse_args()

G = network.Generator(GPU['Generator'])
G = load_model(G, continue_id)

x_hat = G(y_t, e_hat)

save_image(os.path.join(config.GENERATED_DIR, f'Generated_x_hat.png'), x_hat[0])
