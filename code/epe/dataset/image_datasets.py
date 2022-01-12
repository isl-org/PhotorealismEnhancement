import logging
import os
from pathlib import Path
import random

import imageio
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data

from .batch_types import ImageBatch
from .utils import mat2tensor

logger = logging.getLogger(__file__)

class ImageDataset(torch.utils.data.Dataset):
	def __init__(self, name, img_paths, transform=None):
		"""

		name -- Name used for debugging, log messages.
		img_paths - an iterable of paths to individual image files. Only JPG and PNG files will be taken.
		transform -- Transform to be applied to images during loading.
		"""

		img_paths  = [Path(p[0] if type(p) is tuple else p) for p in img_paths]
		self.paths = sorted([p for p in img_paths if p.is_file() and p.suffix in ['.jpg', '.png']])
		
		self._path2id    = {p.stem:i for i,p in enumerate(self.paths)}
		self.transform   = transform
		
		self.name = name
		self._log = logging.getLogger(f'epe.dataset.{name}')
		self._log.info(f'Found {len(self.paths)} images.')
		pass


	def _load_img(self, path):
		try:
			return np.clip(imageio.imread(path).astype(np.float32) / 255.0, 0.0, 1.0)[:,:,:3]
		except:
			logging.exception(f'Failed to load {path}.')
			raise
		pass


	def get_id(self, path):
		return self._path2id.get(Path(path))


	def __getitem__(self, index):
		
		idx  = index % self.__len__()
		path = self.paths[idx]
		img  = self._load_img(path)

		if self.transform is not None:
			img = self.transform(img)
			pass

		img = mat2tensor(img)   
		return ImageBatch(img, path)


	def __len__(self):
		return len(self.paths)
