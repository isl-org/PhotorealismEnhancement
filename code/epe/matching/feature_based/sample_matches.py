import argparse
import csv
from pathlib import Path
import random

from imageio import imwrite
import numpy as np
import torch
from torchvision.utils import make_grid

from epe.dataset import ImageBatch, ImageDataset
from epe.dataset.utils import read_filelist

# for each threshold,
# find a couple of samples
# load them
# make a sample picture

def load_crops(path):

	paths  = []
	coords = []
	with open(path) as file:
		reader = csv.DictReader(file)
		for row in reader:
			paths.append(row['path'])
			coords.append((int(row['r0']), int(row['r1']), int(row['c0']), int(row['c1'])))
			pass
		pass
	return paths, coords


if __name__ == '__main__':

	p = argparse.ArgumentParser("Create a images with sample matches across datasets for several distance thresholds.")
	p.add_argument('src_img_path',  type=Path, help="Path to file with image paths.")
	p.add_argument('src_crop_path', type=Path, help="Path to file with sampled crops.")
	p.add_argument('dst_img_path',  type=Path, help="Path to file with image paths.")
	p.add_argument('dst_crop_path', type=Path, help="Path to file with sampled crops.")
	p.add_argument('match_path', type=Path, help="Path to file with matches.")
	args = p.parse_args()

	src_dataset = ImageDataset('src', read_filelist(args.src_img_path, 1, False))
	dst_dataset = ImageDataset('dst', read_filelist(args.dst_img_path, 1, False))

	data = np.load(args.match_path)
	s = data['dist']#[:,0]

	src_paths, src_coords = load_crops(args.src_crop_path)
	dst_paths, dst_coords = load_crops(args.dst_crop_path)
	
	dst_id = data['ind']
	thresholds = [0.0, 0.1, 0.2,0.5,0.6, 0.7, 1.0,1.2, 1.5, 2.0]
	for ti, t in enumerate(thresholds[1:]): 
	
		print(f'Sampling dist at {t}...')
		src_id, knn = np.nonzero(np.logical_and(thresholds[ti] < s, s < t))
		crops = []
		rd = np.random.permutation(src_id.shape[0])
		for x in range(min(25,src_id.shape[0])):
			i = int(rd[x])
			print(f'\tloading sample {i}...')
			img, _ = src_dataset.get_by_path(src_paths[int(src_id[i])])
			r0,r1,c0,c1 = src_coords[int(src_id[i])]
			a = img[:,r0:r1,c0:c1].unsqueeze(0)
			img, _ = dst_dataset.get_by_path(dst_paths[int(dst_id[int(src_id[i]), int(knn[i])])])
			r0,r1,c0,c1 = dst_coords[int(dst_id[int(src_id[i]), int(knn[i])])]
			b = img[:,r0:r1,c0:c1].unsqueeze(0)
			crops.append(a)
			crops.append(b)
			pass

		if len(crops) > 0:
			grid = make_grid(torch.cat(crops, 0), nrow=2)
			imwrite(f'knn_{t}.jpg', (255.0*grid.permute(1,2,0).numpy()).astype(np.uint8))
			pass
		pass
	pass
