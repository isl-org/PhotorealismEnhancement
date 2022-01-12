import logging
from pathlib import Path

import imageio
import numpy as np
from skimage.transform import resize
import scipy.io as sio
import torch

from .batch_types import EPEBatch
from .synthetic import SyntheticDataset
from .utils import mat2tensor, normalize_dim

def center(x, m, s):
	x[0,:,:] = (x[0,:,:] - m[0]) / s[0]
	x[1,:,:] = (x[1,:,:] - m[1]) / s[1]
	x[2,:,:] = (x[2,:,:] - m[2]) / s[2]
	return x


def material_from_gt_label(gt_labelmap):
	""" Merges several classes. """

	h,w = gt_labelmap.shape
	shader_map = np.zeros((h, w, 12), dtype=np.float32)
	shader_map[:,:,0] = (gt_labelmap == 23).astype(np.float32) # sky
	shader_map[:,:,1] = (np.isin(gt_labelmap, [6, 7, 8, 9, 10])).astype(np.float32) # road / static / sidewalk
	shader_map[:,:,2] = (np.isin(gt_labelmap, [26,27,28,29,30,31,32,33])).astype(np.float32) # vehicle
	shader_map[:,:,3] = (gt_labelmap == 22).astype(np.float32) # terrain
	shader_map[:,:,4] = (gt_labelmap == 21).astype(np.float32) # vegetation
	shader_map[:,:,5] = (np.isin(gt_labelmap, [24,25])).astype(np.float32) # person
	shader_map[:,:,6] = (np.isin(gt_labelmap, [17,18])).astype(np.float32) # infrastructure
	shader_map[:,:,7] = (gt_labelmap == 19).astype(np.float32) # traffic light
	shader_map[:,:,8] = (gt_labelmap == 20).astype(np.float32) # traffic sign
	shader_map[:,:,9] = (gt_labelmap == 1).astype(np.float32) # ego vehicle
	shader_map[:,:,10] = (np.isin(gt_labelmap, [4, 11, 12, 13, 14, 15, 16])).astype(np.float32) # building
	shader_map[:,:,11] = (np.isin(gt_labelmap, [0,5])).astype(np.float32) # unlabeled
	return shader_map


class PfDDataset(SyntheticDataset):
	def __init__(self, paths, transform=None, gbuffers='fake'):
		"""


		paths -- list of tuples with (img_path, robust_label_path, gbuffer_path, gt_label_path)
		"""

		super(PfDDataset, self).__init__('GTA')

		assert gbuffers in ['all', 'img', 'no_light', 'geometry', 'fake']

		self.transform = transform
		self.gbuffers  = gbuffers
		# self.shader    = class_type

		self._paths    = paths
		self._path2id  = {p[0].stem:i for i,p in enumerate(self._paths)}
		if self._log.isEnabledFor(logging.DEBUG):
			self._log.debug(f'Mapping paths to dataset IDs (showing first 30 entries):')
			for i,(k,v) in zip(range(30),self._path2id.items()):
				self._log.debug(f'path2id[{k}] = {v}')
				pass
			pass


		try:
			data = np.load(Path(__file__).parent / 'pfd_stats.npz')
			# self._img_mean  = data['i_m']
			# self._img_std   = data['i_s']
			self._gbuf_mean = data['g_m']
			self._gbuf_std  = data['g_s']
			self._log.info(f'Loaded dataset stats.')
		except:
			# self._img_mean  = None
			# self._img_std   = None
			self._gbuf_mean = None
			self._gbuf_std  = None
			pass

		self._log.info(f'Found {len(self._paths)} samples.')
		pass


	@property
	def num_gbuffer_channels(self):
		""" Number of image channels the provided G-buffers contain."""
		return {'fake':32, 'all':26, 'img':0, 'no_light':17, 'geometry':8}[self.gbuffers]


	@property
	def num_classes(self):
		""" Number of classes in the semantic segmentation maps."""
		return {'fake':12, 'all':12, 'img':0, 'no_light':0, 'geometry':0}[self.gbuffers]


	@property
	def cls2gbuf(self):
		if self.gbuffers == 'all':
			# all: just handle sky class differently
			return {\
				0:lambda g:g[:,15:21,:,:]}
		else:
			return {}


	def get_id(self, img_filename):
		return self._path2id.get(Path(img_filename).stem)


	def __getitem__(self, index):

		index  = index % self.__len__()
		img_path, robust_label_path, gbuffer_path, gt_label_path = self._paths[index]

		if not gbuffer_path.exists():
			self._log.error(f'Gbuffers at {gbuffer_path} do not exist.')
			raise FileNotFoundError
			pass

		data = np.load(gbuffer_path)

		if self.gbuffers == 'fake':
			img       = mat2tensor(imageio.imread(img_path).astype(np.float32) / 255.0)
			gbuffers  = mat2tensor(data['data'].astype(np.float32))
			gt_labels = material_from_gt_label(imageio.imread(gt_label_path))
			if gt_labels.shape[0] != img.shape[-2] or gt_labels.shape[1] != img.shape[-1]:
				gt_labels = resize(gt_labels, (img.shape[-2], img.shape[-1]), anti_aliasing=True, mode='constant')
			gt_labels = mat2tensor(gt_labels)
			pass
		else:
			img       = mat2tensor(data['img'].astype(np.float32) / 255.0)
			gbuffers  = mat2tensor(data['gbuffers'].astype(np.float32))
			gt_labels = mat2tensor(data['shader'].astype(np.float32))
			pass

		if torch.max(gt_labels) > 128:
			gt_labels = gt_labels / 255.0
			pass

		if self._gbuf_mean is not None:
			gbuffers = center(gbuffers, self._gbuf_mean, self._gbuf_std)
			pass

		if not robust_label_path.exists():
			self._log.error(f'Robust labels at {robust_label_path} do not exist.')
			raise FileNotFoundError
			pass

		robust_labels = imageio.imread(robust_label_path)
		robust_labels = torch.LongTensor(robust_labels[:,:]).unsqueeze(0)

		return EPEBatch(img, gbuffers=gbuffers, gt_labels=gt_labels, robust_labels=robust_labels, path=img_path, coords=None)


	def __len__(self):
		return len(self._paths)
