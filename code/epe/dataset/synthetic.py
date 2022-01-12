import logging

import torch.utils.data

class SyntheticDataset(torch.utils.data.Dataset):
	""" Synthetic datasets provide additional information about a scene.

	They may provide image-sized G-buffers, containing geometry, material, 
	or lighting informations, or semantic segmentation maps.

	"""

	def __init__(self, name):
		super(SyntheticDataset, self).__init__()
		self._name     = name
		self._log      = logging.getLogger(f'epe.dataset.{self._name}')
		pass

	@property
	def name(self):
		return self._name

	@property
	def num_gbuffer_channels(self):
		""" Number of image channels the provided G-buffers contain."""
		raise NotimplementedError

	@property
	def num_classes(self):
		""" Number of classes in the semantic segmentation maps."""
		raise NotimplementedError

	@property
	def cls2gbuf(self):
		return NotimplementedError
