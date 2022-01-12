import logging
from math import sqrt

import torch
import torch.nn as nn


logger = logging.getLogger('epe.nf')

norm_factory = {\
	'none': None,
	'group': lambda d: nn.GroupNorm(8,d),
	'batch': lambda d: nn.BatchNorm2d(d, track_running_stats=False),
	'inst':  lambda d: nn.InstanceNorm2d(d, affine=True, track_running_stats=False),
	'domain':lambda d: nn.DomainNorm(d),
}


def make_conv_layer(dims, strides=1, leaky_relu=True, spectral=False, norm_factory=None, skip_final_relu=False, kernel=3):
	""" Make simple convolutional networks without downsampling.

	dims -- list with channel widths, where len(dims)-1 is the number of concolutional layers to create.
	strides -- stride of first convolution if int, else stride of each convolution, respectively
	leaky_relu -- yes or no (=use ReLU instead)
	spectral -- use spectral norm
	norm_factory -- function taking a channel width and returning a normalization layer.
	skip_final_relu -- don't use a relu at the end
	kernel -- width of kernel
	"""

	if type(strides) == int:
		strides = [strides] + [1] * (len(dims)-2)
		pass

	c = nn.Conv2d(dims[0], dims[1], kernel, stride=strides[0], bias=spectral)
	m = [] if kernel == 1 else [nn.ReplicationPad2d(kernel // 2)]
	m += [c if not spectral else torch.nn.utils.spectral_norm(c)]

	if norm_factory:
		m += [norm_factory(dims[1])]
		pass

	m += [nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)]

	num_convs = len(dims)-2
	for i,di in enumerate(dims[2:]):
		c = nn.Conv2d(dims[i+1], di, 3, stride=strides[i+1], bias=spectral)

		if kernel > 1:
			m += [nn.ReplicationPad2d(kernel // 2)]
		m += [c if not spectral else torch.nn.utils.spectral_norm(c)]
		
		if norm_factory:
			m += [norm_factory(di)]
			pass
		
		if i == num_convs-1 and skip_final_relu:
			continue
		else:
			m += [nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)]
		pass

	return nn.Sequential(*m)


class ResBlock(nn.Module):
	def __init__(self, dims, first_stride=1, leaky_relu=True, spectral=False, norm_factory=None, kernel=3):
		super(ResBlock, self).__init__()

		self.conv = make_conv_layer(dims, first_stride, leaky_relu, spectral, norm_factory, True, kernel=kernel)
		self.down = make_conv_layer([dims[0], dims[-1]], first_stride, leaky_relu, spectral, None, True, kernel=kernel) \
			if first_stride != 1 or dims[0] != dims[-1] else None
		self.relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
		pass

	def forward(self, x):
		return self.relu(self.conv(x) + (x if self.down is None else self.down(x)))


class Res2Block(nn.Module):
	def __init__(self, dims, first_stride=1, leaky_relu=True):
		super(Res2Block, self).__init__()

		self.conv = make_conv_layer(dims, first_stride, leaky_relu, False, None, False, kernel=3)
		self.down = make_conv_layer([dims[0], dims[-1]], first_stride, leaky_relu, False, None, True, kernel=1) \
			if first_stride != 1 or dims[0] != dims[-1] else None
		self.relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
		pass

	def forward(self, x):
		return 0.1 * self.conv(x) + (x if self.down is None else self.down(x))


class BottleneckBlock(nn.Module):
	def __init__(self, dim_in, dim_mid, dim_out, stride=1):
		super(BottleneckBlock, self).__init__()
		self._conv1 = nn.Conv2d(dim_in, dim_mid, 1)
		self._conv2 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(dim_mid, dim_mid, 3, stride=stride))
		self._conv3 = nn.Conv2d(dim_mid, dim_out, 1)
		self._relu  = nn.LeakyReLU(0.2, True) if leaky_relu else nn.ReLU(True)
		self._norm1 = nn.GroupNorm(dim_mid)
		self._norm2 = nn.GroupNorm(dim_mid)
		self._norm3 = nn.GroupNorm(dim_out)
		self._down  = nn.Conv2d(dim_in, dim_out, 1, stride=stride) if stride > 1 or dim_in != dim_out else None
		pass

	def forward(self, x):
		r = x if self_down is None else self._down(x)
		x = self._conv1(x)
		x = self._norm1(x)
		x = self._relu(x)
		x = self._conv2(x)
		x = self._norm2(x)
		x = self._relu(x)
		x = self._conv3(x)
		x = self._norm3(x)
		x = x + r			
		x = self._relu(x)
		return x


class ResnextBlock(nn.Module):
	def __init__(self, dim_in, dim_mid, dim_out, groups=8, stride=1):
		super(ResnextBlock, self).__init__()
		self._conv1 = nn.Conv2d(dim_in, dim_mid, 1)
		self._conv2 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(dim_mid, dim_mid, 3, stride=stride, groups=groups))
		self._conv3 = nn.Conv2d(dim_mid, dim_out, 1)
		self._relu  = nn.LeakyReLU(0.2, True) if False else nn.ReLU(True)
		self._norm1 = nn.GroupNorm(groups, dim_mid)
		self._norm2 = nn.GroupNorm(groups, dim_mid)
		self._norm3 = nn.GroupNorm(groups, dim_out)
		self._down  = nn.Conv2d(dim_in, dim_out, 1, stride=stride) if stride > 1 or dim_in != dim_out else None
		pass

	def forward(self, x):
		r = x if self._down is None else self._down(x)
		x = self._conv1(x)
		x = self._norm1(x)
		x = self._relu(x)
		x = self._conv2(x)
		x = self._norm2(x)
		x = self._relu(x)
		x = self._conv3(x)
		x = self._norm3(x)
		x = x + r			
		x = self._relu(x)
		return x


