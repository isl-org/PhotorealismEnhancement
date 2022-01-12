import logging

import torch
import torch.nn as nn

import epe.network.network_factory as nf


logger = logging.getLogger('epe.network.gb_encoder')

_gbuffer_class_encoder_factory = {\
	'relu':    lambda di,do, s0, s1: nf.make_conv_layer([di,64,do], [s0,s1], False, False, None),
	'lrelu':   lambda di,do, s0, s1: nf.make_conv_layer([di,64,do], [s0,s1], True,  False, None),
	'spectral':lambda di,do, s0, s1: nf.make_conv_layer([di,64,do], [s0,s1], True,  True,  None),
	'group':   lambda di,do, s0, s1: nf.make_conv_layer([di,64,do], [s0,s1], True, False,  nf.norm_factory['group']),
	'batch':   lambda di,do, s0, s1: nf.make_conv_layer([di,64,do], [s0,s1], True,  False, nf.norm_factory['batch']),
	'domain':  lambda di,do, s0, s1: nf.make_conv_layer([di,64,do], [s0,s1], True,  False, nf.norm_factory['domain']),
	'residual':lambda di,do, s0, s1: nn.Sequential(nf.ResBlock([di, 64, 64], s0, False, True), nf.ResBlock([64, 64, do], s1, False, True)),
	'residual2':lambda di,do, s0, s1: nn.Sequential(nf.Res2Block([di, 64, 64], s0), nf.Res2Block([64, 64, do], s1)),
}

di2rnp = {16:[1,4], 32:[2,4], 64:[2,4], 128:[2,4], 256:[2,4], 512:[4,8]}
_gbuffer_joint_encoder_factory = {\
	'relu':    lambda di, do, s: nf.make_conv_layer([di, do, do], s, False, False, None),
	'lrelu':   lambda di, do, s: nf.make_conv_layer([di, do, do], s, True,  False, None),
	'brelu':   lambda di, do, s: nf.make_conv_layer([di, do, do], s, False, False, (lambda d:nn.BatchNorm2d(d, track_running_stats=False))),
	'spectral':lambda di, do, s: nf.make_conv_layer([di, do, do], s, True,  True,  None),
	'group':   lambda di, do, s: nf.make_conv_layer([di, do, do], s, True,  False, (lambda d:nn.GroupNorm(8,d))),
	'batch':   lambda di, do, s: nf.make_conv_layer([di, do, do], s, True,  False, (lambda d:nn.BatchNorm2d(d, track_running_stats=False))),
	'domain':  lambda di, do, s: nf.make_conv_layer([di, do, do], s, True,  False, (lambda d:DomainNorm(d))),	
	'residual':lambda di, do, s: nn.Sequential(nf.ResBlock([di, do, do], s, False, True), nf.ResBlock([do, do, do], 1, False, True)),
	'residual2':lambda di, do, s: nn.Sequential(nf.Res2Block([di, do, do], s), nf.Res2Block([do, do, do], 1)),
	'resnext':lambda di,do,s: nf.ResnextBlock(di, di//di2rnp[di][0], do, groups=di2rnp[di][1], stride=s),
}


def _append_downsampled_gbuffers(g_list, x_list):
	""" Dynamically downsample G-buffers, matching resolution in feature maps."""

	for i in range(len(g_list), len(x_list)):
		g_list.append(torch.nn.functional.interpolate(g_list[i-1], size=[x_list[i].shape[-2],x_list[i].shape[-1]], mode='bilinear', align_corners=False))
		pass
	return g_list


def _append_downsampled_shaders(s, s_list, x_list):
	""" Dynamically downsample G-buffers, matching resolution in feature maps."""

	if s.shape[1] != 1:
		for i in range(len(s_list), len(x_list)):
			s_list.append(torch.argmax(torch.nn.functional.interpolate(s, size=[x_list[i].shape[-2],x_list[i].shape[-1]], mode='bilinear', align_corners=False), dim=1, keepdims=True).long())
			pass
	else:
		for i in range(len(s_list), len(x_list)):
			s_list.append(torch.nn.functional.interpolate(s, size=[x_list[i].shape[-2],x_list[i].shape[-1]], mode='nearest').long())
			pass
		pass

	return s_list


class GBufferEncoder(nn.Module):
	def __init__(self, num_down_levels, gbuffer_norm, num_classes, num_gbuffer_channels, cls2gbuf, num_branches):
		"""

		num_down_levels -- number of initial downsample levels, same as the stem in HRNet
		cls2gbuf -- Dictionary with functions that collect gbuffers for specific classes. 
		"""
		super(GBufferEncoder, self).__init__()

		self._log = logging.getLogger('epe.network.gb_encoder')
		self._log.debug(f'Creating G-bufferEncoder with {gbuffer_norm} norm for {num_classes} classes, {num_gbuffer_channels} G-buffers and {num_branches} branches.')

		self.num_classes          = num_classes
		self.num_gbuffer_channels = num_gbuffer_channels
		self.cls2gbuf             = cls2gbuf
		self.num_branches         = num_branches
		self.norm_type            = gbuffer_norm
		self.num_down_levels      = num_down_levels

		self.class_encoders, self.joint_encoder_layers = self._make_gbuffer_encoders()
		pass


	def _compute_gbuf_encoder_dim(self):
		""" Compute number of input channels required for each class. """

		t = torch.zeros(1,self.num_gbuffer_channels, 1, 1)
		return {c:f(t).shape[1] for c,f in self.cls2gbuf.items()}


	def _make_gbuffer_encoders(self):
		cls2dim = self._compute_gbuf_encoder_dim()

		s0 = 1 if self.num_down_levels < 1 else 2
		s1 = 1 if self.num_down_levels < 2 else 2
		class_encoders = []
		for i in range(self.num_classes):
			dim_in = self.num_gbuffer_channels if not i in cls2dim else cls2dim[i]
			class_encoders.append(_gbuffer_class_encoder_factory[self.norm_type](dim_in, 128, s0, s1))
			pass

		self._log.debug(f'  Creating joint encoder for {self.num_branches} branches:')
		joint_enoders = []
		for i in range(self.num_branches):
			je = _gbuffer_joint_encoder_factory[self.norm_type](128, 128, 1 if i==0 else 2)
			self._log.debug(f'  {i}: {je}')
			joint_enoders.append(je)
			pass

		return nn.ModuleList(class_encoders), nn.ModuleList(joint_enoders)


	def forward(self, gbuffers, classmap):
		""" Encode G-buffers depending on semantic class.

		gbuffers -- 4D tensor with multiple G-buffers stacked along dim 1.
		classmap -- 4D tensor with one-hot-encoded masks for classes, also stacked along dim 1.

		The cls2gbuf allows easily excluding G-buffers for some classes. This makes sense
		if it is a priori known that G-buffers provide no useful info for this class and 
		prevents learning spurious features from noise.

		Depending on the classmap and cls2gbuf function, G-buffer info for each pixel 
		are routed through different G-buffer Encoder networks. Then, the encoded 
		features are simply merged via the classmaps and encoded for different branches
		of the image enhancment network.
		"""

		if self._log.isEnabledFor(logging.DEBUG):
			self._log.debug(f'G-BufferEncoder:forward(g:{gbuffers.shape}, c:{classmap.shape})')
			pass

		num_classes = classmap.shape[1]
		features = 0
		for c in range(num_classes):
			features += classmap[:,c,:,:] * self.class_encoders[c](\
				self.cls2gbuf[c](gbuffers) if c in self.cls2gbuf else gbuffers)
			pass

		features = [features]
		for layer in self.joint_encoder_layers:
			features.append(layer(features[-1]))
			pass

		return features[1:]


base_norm_factory = {
	'group':lambda d:nn.GroupNorm(8, d, affine=False),
	'batch':lambda d:nn.BatchNorm2d(d, affine=False, track_running_stats=False),
	'inst':	lambda d:nn.InstanceNorm2d(d, affine=False),
}

base_layer_factory = {
	'convr':   lambda di,do:nf.make_conv_layer([di,do],1,False,False),
	# 'residual':lambda di,do:nf.ResBlock([di,do],1,False,False, nf.norm_factory['batch']),
	'residual':lambda di,do:nf.ResBlock([di,do],1,False, True, None),
	'residual2':lambda di,do:nf.Res2Block([di,do,do],1,),
	'resnext':lambda di,do: nf.ResnextBlock(di, di//di2rnp[di][0], do, groups=di2rnp[di][1], stride=1)
}


class BatchNormWrapper(nn.BatchNorm2d):
	def __init__(self, *args, **kwargs):
		super(BatchNormWrapper, self).__init__(*args, **kwargs)

	def forward(self, x, _):
		return super().forward(x)


def gbuffer_norm_factory(name, num_layers):
	if name == 'Default':
		return lambda dim_x: BatchNormWrapper(dim_x, affine=True)
	elif name == 'SPADE':
		return lambda dim_x: GBufferNorm(dim_x, 128, 128, base_norm_factory['batch'], base_layer_factory['convr'],    num_layers=1)
	elif name == 'RAD':
		return lambda dim_x: GBufferNorm(dim_x, 128, 128, base_norm_factory['group'], base_layer_factory['residual'], num_layers=num_layers)
	elif name == 'RNAD':
		return lambda dim_x: GBufferNorm(dim_x, dim_x, dim_x, base_norm_factory['group'], base_layer_factory['residual'], num_layers=num_layers)
	elif name == 'RAC':
		return lambda dim_x: GBufferConv(dim_x, 128, 128, base_layer_factory['residual2'], num_layers=num_layers)
	else:
		raise NotImplementedError


class GBufferNorm(nn.Module):
	""" Adapter for replacing the default normalization layers.	"""

	def __init__(self, dim_x, dim_g, dim_e, norm_func, gbuf_proc, num_layers=1):
		""" Construct a regular normalization layer or G-buffer-dependent normalization.

		dim_x -- channel width for image features
		dim_g -- number of G-buffer channels
		dim_e -- number of channels for internal embedding of G-buffers
		norm_func -- function that takes a channel width and returns a normalization layer
		gbuf_proc -- function that takes a channel width and returns a processing block (e.g., conv layer or residual block) for the G-buffers
		num_layers -- how many of the processing blocks to generate.	
		"""

		super(GBufferNorm, self).__init__()

		self._norm  = norm_func(dim_x) # for image features		
		# self._scale_u = nn.Parameter(torch.zeros(1,1,1,1))

		model = []
		dim_in = dim_g
		for i in range(num_layers):
			model += [gbuf_proc(dim_in,dim_e)]
			dim_in = dim_e
			pass
		self._conv  = nn.Sequential(*model) if num_layers > 0 else None
		self._scale = nn.Conv2d(dim_e,dim_x,1)
		self._bias  = nn.Conv2d(dim_e,dim_x,1)
		pass


	def forward(self, x, g):
		if self._conv is not None:
			g = self._conv(g)
			pass

		return self._norm(x) * self._scale(g) + self._bias(g)


class GBufferConv(nn.Module):
	""" Adapter for replacing the default normalization layers.	"""

	def __init__(self, dim_x, dim_g, dim_e, gbuf_proc, num_layers=1):
		""" Construct a regular normalization layer or G-buffer-dependent normalization.

		dim_x -- channel width for image features
		dim_g -- number of G-buffer channels
		dim_e -- number of channels for internal embedding of G-buffers
		gbuf_proc -- function that takes a channel width and returns a processing block (e.g., conv layer or residual block) for the G-buffers
		num_layers -- how many of the processing blocks to generate.	
		"""

		super(GBufferConv, self).__init__()

		model = []
		dim_in = dim_g
		for i in range(num_layers):
			model += [gbuf_proc(dim_in,dim_e)]
			dim_in = dim_e
			pass
		self._conv  = nn.Sequential(*model) if num_layers > 0 else None
		self._scale = nn.Conv2d(dim_e,dim_x,1)
		self._bias  = nn.Conv2d(dim_e,dim_x,1)
		pass


	def forward(self, x, g):
		if self._conv is not None:
			g = self._conv(g)
			pass

		return x  * (1- 0.1 * self._scale(g)) + 0.1 * self._bias(g)
