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

    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help="Name of the dataset.")
    parser.add_argument('img_list', type=Path, help="Path to csv file with path to images in first column.")
    parser.add_argument('-n', '--num_loaders', type=int, default=1)
    parser.add_argument('-c', '--num_crops', type=int, help="Number of crops to sample per image. Default = 15.", default=15)
    parser.add_argument('--out_dir', type=Path, help="Where to store the crop info.", default='.')
    args = parser.parse_args()

    network   = VGG16(False, padding='none').to(device)
    extract   = lambda img: network.fw_relu(img, 13)[-1]
    crop_size = 196 # VGG-16 receptive field at relu 5-3
    dim       = 512 # channel width of VGG-16 at relu 5-3
    num_crops = args.num_crops

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

    features = np.zeros((len(dataset) * num_crops, dim), np.float16)

    print('Sampling crops...')

    ip = 0
    with open(args.out_dir / f'crop_{args.name}.csv', 'w') as log:
        log.write('id,path,r0,r1,c0,c1\n')
        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader)):
                
                n,_,h,w = batch.img.shape
                assert n == 1

                if i == 0:
                    print(f'Image size is {h}x{w} - sampling {num_crops} crops per image.')
                    pass

                c0s = torch.randint(w-crop_size+1, (num_crops,1))
                r0s = torch.randint(h-crop_size+1, (num_crops,1))

                samples = []
                for j in range(num_crops):
                    r0 = r0s[j].item()
                    c0 = c0s[j].item()
                    r1 = r0 + crop_size
                    c1 = c0 + crop_size
                    samples.append(batch.img[0,:,r0:r1,c0:c1].reshape(1,3,crop_size,crop_size))
                    log.write(f'{ip},{batch.path[0]},{r0},{r1},{c0},{c1}\n')
                    ip += 1
                    pass

                samples = torch.cat(samples, 0)
                samples = samples.to(device, non_blocking=True)
                f = extract(samples)
                
                features[ip-num_crops:ip,:] = f.cpu().numpy().astype(np.float16).reshape(num_crops, dim)
                pass
            pass
        pass

    print('Saving features.')
    np.savez_compressed(args.out_dir / f'crop_{args.name}', crops=features)
    pass
