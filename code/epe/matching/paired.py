import csv
import logging
from pathlib import Path
import random

import numpy as np
import torch

from epe.dataset.batch_types import JointEPEBatch
from epe.dataset.utils import load_crops
from epe.matching import load_matching_crops


class PairedDataset(torch.utils.data.Dataset):
	def __init__(self, source_dataset, target_dataset):
		self._source_dataset = source_dataset
		self._target_dataset = target_dataset

		self.src_crops = []
		self.dst_crops = []

		self._log = logging.getLogger('epe.dataset.PairedDataset')
		pass


	def _get_cropped_items(self, idx, jdx):
		s = self.src_crops[idx]
		t = self.dst_crops[jdx]

		self._log.debug(f'_get_cropped_items:')

		src_id = self._source_dataset.get_id(s[0])
		if src_id is None:
			self._log.debug(f'  src[{idx}](?|{s[1:]}) : {s[0]} does not exist.')
			raise KeyError
		self._log.debug(f'  src[{idx}]({src_id}|{s[1:]}) : {s[0]}')

		dst_id = self._target_dataset.get_id(t[0])
		if dst_id is None:
			self._log.debug(f'  dst[{jdx}](?|{t[1:]}) : {t[0]} does not exist.')
			raise KeyError
		self._log.debug(f'  dst[{idx}]({dst_id}|{t[1:]}) : {t[0]}')

		return JointEPEBatch(self._source_dataset[src_id].crop(*s[1:]), self._target_dataset[dst_id].crop(*t[1:]))


	def __len__(self):
		return len(self.src_crops)


	@property
	def source(self):
		return self._source_dataset


	@property
	def target(self):
		return self._target_dataset


class MatchedCrops(PairedDataset):
	def __init__(self, source_dataset, target_dataset, cfg):

		super(MatchedCrops, self).__init__(source_dataset, target_dataset)

		self._log = logging.getLogger('epe.dataset.MatchedCrops')
		self._log.debug(f'Initializing sampling with matching crops ...')
		self._log.debug(f'  src         : {source_dataset.name}')
		self._log.debug(f'  dst         : {target_dataset.name}')

		matched_crop_path = Path(cfg.get('matched_crop_path', None))
		crop_weight_path  = cfg.get('crop_weight_path', None)

		self._weighted = False

		self.src_crops, self.dst_crops = load_matching_crops(matched_crop_path)

		valid_src_crops, valid_dst_crops = [], []
		valid_ids = []
		for i, (sc, dc) in enumerate(zip(self.src_crops, self.dst_crops)):
			if self._source_dataset.get_id(sc[0]) is not None:
				valid_src_crops.append(sc)
				valid_dst_crops.append(dc)
				valid_ids.append(i)
				pass
			pass

		self._log.debug(f'Done to {len(valid_ids)} crops.')

		self.src_crops = valid_src_crops
		self.dst_crops = valid_dst_crops
		if crop_weight_path is not None:
			d = np.load(crop_weight_path)
			w = d['w']
			w = w[valid_ids]
			self._cumsum = np.cumsum(w) / np.sum(w)
			assert len(self.src_crops) == self._cumsum.shape[0], f'Weights ({self._cumsum.shape[0]}) and source crops ({len(self.src_crops)}) do not match.'
			self._weighted = True
			pass

		self._log.debug('Sampling Initialized.')
		pass

	def __getitem__(self, idx):
		try:
			if self._weighted:
				p   = random.random()
				idx = np.min(np.nonzero(p<self._cumsum)[0])
				pass
			return self._get_cropped_items(idx, idx)
		except KeyError:
			return self.__getitem__(random.randint(0, len(self.src_crops)-1))

	def __len__(self):
		return len(self.src_crops)


class IndependentCrops(PairedDataset):
	def __init__(self, source_dataset, target_dataset, cfg):
		super(IndependentCrops, self).__init__(source_dataset, target_dataset)

		self._crop_size = int(cfg.get('crop_size', 196))
		pass

	def _sample_crop(self, batch):
		r1 = random.randint(self._crop_size, batch.img.shape[-2])
		r0 = r1 - self._crop_size
		c1 = random.randint(self._crop_size, batch.img.shape[-1])
		c0 = c1 - self._crop_size
		return batch.crop(r0, r1, c0, c1)


	def __getitem__(self, idx):
		return self._sample_crop(self._source_dataset[idx]), \
			self._sample_crop(self._target_dataset[random.randint(0, len(self._target_dataset)-1)])


	def __len__(self):
		return len(self._source_dataset)

		
