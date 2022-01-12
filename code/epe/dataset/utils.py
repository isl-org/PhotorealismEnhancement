import csv
import logging
from pathlib import Path

import numpy as np
from skimage.transform import rescale
import torch
from tqdm import tqdm

logger = logging.getLogger('epe.dataset.utils')


def read_filelist(path_to_filelist, num_expected_entries_per_row, check_if_exists=True):
	""" Loads a file with paths to multiple files per row.

	path_to_filelist -- path to text file
	num_expected_entries_per_row -- number of expected entries per row.
	check_if_exists -- checks each path.
	"""

	paths = []
	num_skipped = 0
	with open(path_to_filelist) as file:
		for i, line in enumerate(file):
			t = line.strip().split(',')
			assert len(t) >= num_expected_entries_per_row, \
				f'Expected at least {num_expected_entries_per_row} entries per line. Got {len(t)} instead in line {i} of {path_to_filelist}.'

			ps = [Path(p) for p in t[:num_expected_entries_per_row]]

			if check_if_exists:
				skip = [p for p in ps if not p.exists()]
				if skip:
					num_skipped += 1
					# logger.warn(f'Skipping {i}: {skip[0]} does not exist.')
					continue
					# assert p.exists(), f'Path {p} does not exist.'
					
					pass
				pass

			paths.append(tuple(ps))
			pass
		pass

	if num_skipped > 0:
		logger.warn(f'Skipped {num_skipped} entries since at least one file was missing.')

	return paths


def load_crops(path):
	""" Load crop info from a csv file.

	The file is expected to have columns path,r0,r1,c0,c1
	path -- Path to image
	r0 -- top y coordinate
	r1 -- bottom y coordinate
	c0 -- left x coordinate
	c1 -- right x coordinate
	"""

	path = Path(path)

	if not path.exists():
		logger.warn(f'Failed to load crops from {path} because it does not exist.')
		return []

	crops = []		
	with open(path) as file:
		reader = csv.DictReader(file)
		for row in tqdm(reader):
			crops.append((row['path'], int(row['r0']), int(row['r1']), int(row['c0']), int(row['c1'])))
			pass
		pass
	
	logger.debug(f'Loaded {len(crops)} crops.')
	return crops


def mat2tensor(mat):    
	t = torch.from_numpy(mat).float()
	if mat.ndim == 2:
		return t.unsqueeze(2).permute(2,0,1)
	elif mat.ndim == 3:
		return t.permute(2,0,1)

def normalize_dim(a, d):
	""" Normalize a along dimension d."""
	return a.mul(a.pow(2).sum(dim=d,keepdim=True).clamp(min=0.00001).rsqrt())


def transform_identity(img):
	return img

def make_scale_transform(scale):
	return lambda img: rescale(img, scale, preserve_range=True, anti_aliasing=True, multichannel=True)

def make_scale_transform_w(target_width):
	return lambda img: rescale(img, float(target_width) / img.shape[1], preserve_range=True, anti_aliasing=True, multichannel=True)

def make_scale_transform_h(target_height):
	return lambda img: rescale(img, float(target_height) / img.shape[0], preserve_range=True, anti_aliasing=True, multichannel=True)
