import os, argparse
import numpy as np
from PIL import Image
from models import *
import torch
import torch.nn as nn
import torchvision.transforms as tfs 
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

abs = os.getcwd() + '/'

def tensorShow(tensors, titles=['haze']):
    fig = plt.figure()
    for tensor, tit, i in zip(tensors, titles, range(len(tensors))):
        img = make_grid(tensor)
        npimg = img.numpy()
        ax = fig.add_subplot(221 + i)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(tit)
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default=None,
                    help='its / ots / nh / nh4k (if None: test all models)')
parser.add_argument('--test_imgs', type=str, default='test_imgs',
                    help='Test images folder')
opt = parser.parse_args()

gps = 3
blocks = 19
img_dir = abs + opt.test_imgs + '/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Collect model paths
model_paths = []
model_root = abs + 'trained_models/'

if opt.task is None:
    print("No task specified: testing all models found in trained_models/")

    for f in os.listdir(model_root):
        if f.endswith('.pk'):
            model_paths.append(os.path.join(model_root, f))

    if not model_paths:
        raise FileNotFoundError("No .pk models found in trained_models/")
else:
    model_paths = [
        abs + f"trained_models/{opt.task}_train_ffa_{gps}_{blocks}.pk"
    ]
 
print('If you want to see the next dehazed result, please click the \'X\' button in the screen.')
# Run testing for each model
for model_path in model_paths:
    model_name = os.path.basename(model_path).replace('.pk', '')

    output_dir = abs + f"pred_FFA_{model_name}/"
    print("pred_dir:", output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print(f"\nTesting model: {model_name}")

    ckp = torch.load(model_path, map_location=device, weights_only=False)
    net = FFA(gps=gps, blocks=blocks)
    net = nn.DataParallel(net)
    net.load_state_dict(ckp['model'])
    net.eval()

    for im in os.listdir(img_dir):
        print(f'\r {im}', end='', flush=True)

        haze = Image.open(img_dir + im)

        haze1 = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.64, 0.6, 0.58],
                          std=[0.14, 0.15, 0.152])
        ])(haze)[None, ::]

        haze_no = tfs.ToTensor()(haze)[None, ::]

        with torch.no_grad():
            pred = net(haze1)

        ts = torch.squeeze(pred.clamp(0, 1).cpu())
        tensorShow([haze_no, pred.clamp(0, 1).cpu()], ['haze', 'pred'])
        vutils.save_image(ts, output_dir + im.split('.')[0] + '_FFA.png')

print("\nDone!")