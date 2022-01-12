from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm

from epe.matching import load_matching_crops

p = ArgumentParser()
p.add_argument('matched_crop_path', type=Path, help="Path to csv with matched crop info.")
p.add_argument('height', type=int, help="Height of images in dataset.")
p.add_argument('width', type=int, help="Width of images in dataset.")
p.add_argument('weight_path', type=Path, help="Path to weight file.")
args = p.parse_args()

src_crops,_ = load_matching_crops(args.matched_crop_path)

d = np.zeros((args.height, args.width), dtype=np.int32)
print('Computing density...')
for s in tqdm(src_crops): 
	d[s[1]:s[2],s[3]:s[4]] += 1 

print('Computing individual weights...')
w = np.zeros((len(src_crops), 1)) 
for i, s in enumerate(tqdm(src_crops)):
	w[i,0] = np.mean(d[s[1]:s[2],s[3]:s[4]])
	pass

N = np.max(d)
p = N / w
np.savez_compressed(args.weight_path, w=p)
