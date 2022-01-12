import argparse
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from epe.dataset import ImageBatch, ImageDataset
from epe.dataset.utils import read_filelist
from epe.network import VGG16


def seed_worker(id):
	np.random.seed(torch.initial_seed() % np.iinfo(np.int32).max)
	pass

if __name__ == '__main__':

    device = torch.device('cuda')

    parser = argparse.ArgumentParser("Compute a fake set of G-buffers using VGG-16 features. \
This is just to showcase the pipeline/network architecture. \
Instead of these fake G-buffers generated from images, we strongly recommend extracting suitable info from the rendering pipeline.")
    parser.add_argument('name', type=str, help="Name of the dataset.")
    parser.add_argument('img_list', type=Path, help="Path to csv file with path to images in first column.")
    parser.add_argument('-n', '--num_loaders', type=int, default=1)
    parser.add_argument('--out_dir', type=Path, help="Where to store the fake gbuffer.", default='.')
    args = parser.parse_args()

    network   = VGG16(False, padding='none').to(device)

    def extract(img):
        f = network.fw_relu(img, 3)[-1]
        return network.relu_3[1](network.relu_3[0](f))

    #extract   = lambda img: network.fw_relu(img, 13)[-1]
    crop_size = 196 # VGG-16 receptive field at relu 5-3
    dim       = 512 # channel width of VGG-16 at relu 5-3

    dataset = ImageDataset(args.name, read_filelist(args.img_list, 1, False))

    # compute mean/std

    loader  = torch.utils.data.DataLoader(dataset, \
        batch_size=1, shuffle=True, \
        num_workers=args.num_loaders, pin_memory=True, drop_last=False, worker_init_fn=seed_worker, collate_fn=ImageBatch.collate_fn)

    print('Computing mean/std...')

    m, s = [], []
    for i, batch in tqdm(zip(range(1000), loader)):
        m.append(batch.img.mean(dim=(2,3)))
        s.append(batch.img.std(dim=(2,3)))
        pass

    m = torch.cat(m, 0).mean(dim=0)
    s = torch.cat(s, 0).mean(dim=0)

    network.set_mean_std(m[0], m[1], m[2], s[0], s[1], s[2])

    loader  = torch.utils.data.DataLoader(dataset, \
        batch_size=1, shuffle=False, \
        num_workers=args.num_loaders, pin_memory=True, drop_last=False, worker_init_fn=seed_worker, collate_fn=ImageBatch.collate_fn)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):

            n,_,h,w = batch.img.shape
            assert n == 1

            img = batch.img.to(device, non_blocking=True)
            f = extract(img)
            f = f.reshape(1,32,4,f.shape[-2],f.shape[-1]).mean(dim=2)
            f = torch.nn.functional.interpolate(f, size=(h,w), mode='bicubic')
            f = f.reshape(32,h,w).permute(1,2,0).cpu().numpy().astype(np.float16)
            #print(batch.path)
            np.savez_compressed(args.out_dir / f'{batch.path[0].stem}', data=f)
            pass
        pass
    pass
