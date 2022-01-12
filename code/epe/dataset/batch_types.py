import logging

import torch

logger = logging.getLogger('epe.dataset.batch_types')


def _safe_to(a, device):
	return a.to(device, non_blocking=True) if a is not None else None

def _safe_expand(a):
	return a if a is None or a.dim() == 4 else a.unsqueeze(0)

def _safe_cat(s, dim):
	try: 
		return torch.cat(s, dim)
	except TypeError:
		return None


class Batch:
	def to(self, device):
		""" Move all internal tensors to specified device. """
		raise NotImplementedError

class ImageBatch(Batch):
	""" Augment an image tensor with identifying info like path and crop coordinates. 

	img  -- RGB image
	path -- Path to image
	coords -- Crop coordinates representing the patch stored in img and taken from the path.

	The coords are used for keeping track of the image position for cropping. If we load an image
	and crop part of it, we want to still be able to compute the correct coordinates for the original
	image. That's why we store the coordinates used for cropping (top y, bottom y, left x, right x).
	"""

	def __init__(self, img, path=None, coords=None):
		self.img      = _safe_expand(img)
		self.path     = path
		self._coords  = (0, img.shape[-2], 0, img.shape[-1]) if coords is None else coords
		pass

	def to(self, device):
		return ImageBatch(_safe_to(self.img, device), path=self.path)

	def _make_new_crop_coords(self, r0, r1, c0, c1):
		return (self._coords[0]+r0, self._coords[0]+r1, self._coords[2]+c0, self._coords[2]+c1)

	def crop(self, r0, r1, c0, c1):
		""" Return cropped patch from image tensor(s). """
		coords = self._make_new_crop_coords(r0, r1, c0, c1)
		return ImageBatch(self.img[:,:,r0:r1,c0:c1], path=self.path, coords=coords)

	@classmethod
	def collate_fn(cls, samples):
		imgs          = _safe_cat([s.img for s in samples], 0)
		paths         = [s.path for s in samples]
		return ImageBatch(imgs, path=paths)
	pass


class EPEBatch(ImageBatch):
	def __init__(self, img, gbuffers=None, gt_labels=None, robust_labels=None, path=None, coords=None):
		""" Collect all input info for a network.

		img           -- RGB image
		gbuffers      -- multi-channel image with additional scene info (e.g., depth, surface normals, albedo)
		gt_labels     -- semantic segmentation provided by synthetic dataset
		robust_labels -- semantic segmentation by robust pretrained method (e.g., MSeg)		
		path          -- Path to image
		coords        -- Crop coordinates that represent the image patch.
		"""

		super(EPEBatch, self).__init__(img, path, coords)

		self.gt_labels     = _safe_expand(gt_labels)
		self.gbuffers      = _safe_expand(gbuffers)
		self.robust_labels = _safe_expand(robust_labels)
		pass


	@property
	def imggbuf(self):
		return torch.cat((self.img, self.gbuffers), 1)


	def to(self, device):
		return EPEBatch(_safe_to(self.img, device), 
			gbuffers=_safe_to(self.gbuffers, device), 
			gt_labels=_safe_to(self.gt_labels, device),
			robust_labels=_safe_to(self.robust_labels, device), path=self.path)


	def crop(self, r0, r1, c0, c1):
		""" Crop all images in the batch.

		"""

		# if self.labelmap is not None:
		# 	if self.labelmap.shape[2] == self.img.shape[2] and self.labelmap.shape[3] == self.img.shape[3]:
		# 		labelmap = self.labelmap[:,:,r0:r1,c0:c1]
		# 	else:
		# 		labelmap = None
		# 		pass
		# 	pass
		# else:
		# 	labelmap = self.labelmap
		# 	pass

		gbuffers      = None if self.gbuffers is None else self.gbuffers[:,:,r0:r1,c0:c1]
		gt_labels     = None if self.gt_labels is None else self.gt_labels[:,:,r0:r1,c0:c1]		
		robust_labels = None if self.robust_labels is None else self.robust_labels[:,:,r0:r1,c0:c1]		
		coords        = self._make_new_crop_coords(r0, r1, c0, c1)
		return EPEBatch(self.img[:,:,r0:r1,c0:c1], \
			gbuffers=gbuffers, gt_labels=gt_labels, robust_labels=robust_labels, 
			path=self.path, coords=coords)

	@classmethod
	def collate_fn(cls, samples):
		imgs          = _safe_cat([s.img for s in samples], 0)
		gbuffers      = _safe_cat([s.gbuffers for s in samples], 0)
		robust_labels = _safe_cat([s.robust_labels for s in samples], 0)
		gt_labels     = _safe_cat([s.gt_labels for s in samples], 0)
		paths         = [s.path for s in samples]
		return EPEBatch(imgs, gbuffers=gbuffers, gt_labels=gt_labels, robust_labels=robust_labels, 
			path=paths)
	pass


class JointEPEBatch(Batch):
	""" Combines two batches into one. """

	def __init__(self, fake, real):
		self.real = real
		self.fake = fake

	def to(self, device):
		return JointEPEBatch(self.fake.to(device), self.real.to(device))

	@classmethod
	def collate_fn(cls, samples):
		reals = [s.real for s in samples]
		fakes = [s.fake for s in samples]
		return JointEPEBatch(EPEBatch.collate_fn(fakes), EPEBatch.collate_fn(reals))