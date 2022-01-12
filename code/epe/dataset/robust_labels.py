import logging
from pathlib import Path

import imageio
import torch

from .batch_types import EPEBatch
from .image_datasets import ImageDataset
from .utils import mat2tensor

logger = logging.getLogger('epe.dataset.robust')

class RobustlyLabeledDataset(ImageDataset):
	def __init__(self, name, img_and_robust_label_paths, img_transform=None, label_transform=None):
		""" Create an image dataset with robust labels.

		name -- Name of dataset, used for debug output and finding corresponding sampling strategy
		img_and_robust_label_paths -- Iterable of tuple containing image path and corresponding path to robust label map. Assumes that filenames are unique!
		img_transform -- Transform (func) to apply to image during loading
		label_transform -- Transform (func) to apply to robust label map during loading
		"""
		self._log = logging.getLogger(f'epe.dataset.{name}')

		self._img2label = {}
		for img_path,lab_path in img_and_robust_label_paths:
			img_path = Path(img_path)
			lab_path = Path(lab_path)

			if img_path.is_file() and img_path.suffix in ['.jpg', '.png'] and \
				lab_path.is_file() and lab_path.suffix == '.png':
				self._img2label[img_path] = lab_path
				pass
			pass

		self.paths           = sorted(self._img2label.keys())
		self._path2id        = {p.stem:i for i,p in enumerate(self.paths)}
		self.transform       = img_transform
		self.label_transform = label_transform
		self.name            = name

		self._log.info(f'Found {len(self.paths)} images.')
		if len(self.paths) < 1:
			self._log.warn('Dataset is empty!')
			pass
		pass


	def get_id(self, img_filename):
		""" Get dataset ID for sample given img_filename."""
		return self._path2id.get(Path(img_filename).stem)


	def __getitem__(self, index):
		
		idx      = index % self.__len__()
		img_path = self.paths[idx]
		img      = self._load_img(img_path)

		if self.transform is not None:
			img = self.transform(img)
			pass

		img = mat2tensor(img)

		label_path    = self._img2label[img_path]
		robust_labels = imageio.imread(label_path)

		if self.label_transform is not None:
			robust_labels = self.label_transform(robust_labels)
			pass

		robust_labels = torch.LongTensor(robust_labels).unsqueeze(0)

		return EPEBatch(img, path=img_path, robust_labels=robust_labels)
	pass

