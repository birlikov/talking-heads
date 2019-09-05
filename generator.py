import argparse
import os
import pickle
from run import load_model, save_image

import config


parser = argparse.ArgumentParser(description='Generate frame from landmark')
parser.add_argument("--continue_id", required=True, help='continue_id of the generator model')
parser.add_argument("--e_hat", required=True, help='path to e_hat tensor')
parser.add_argument("--landmark", required=True, help = 'path to landmark tensor')
args = parser.parse_args()

#
# def load_model(model, continue_id):
#     filename = f'{type(model).__name__}_{continue_id}.pth'
#     state_dict = torch.load(os.path.join(config.MODELS_DIR, filename))
#     model.load_state_dict(state_dict)
#     return model
#
# def save_image(filename, data):
#     if not os.path.isdir(config.GENERATED_DIR):
#         os.makedirs(config.GENERATED_DIR)
#
#     data = data.clone().detach().cpu()
#     img = (data.numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
#     img = Image.fromarray(img)
#     img.save(filename)

G = load_model(G, continue_id)

x_hat = G(y_t, e_hat)

save_image(os.path.join(config.GENERATED_DIR, f'Generated_x_hat.png'), x_hat[0])
