# ------------------------------------------------------------------------------
# Original HRNet:
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
# Modifications to ImageEnhancementNetwork
# Copyright (c) Intel
# Licensed under the MIT License.
# Written by Stephan Richter (stephan.richter@intel.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum
import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

import epe.network.gb_encoder as ge
import epe.network.network_factory as nf

# from .sync_bn.inplace_abn.bn import InPlaceABNSync

BatchNorm2d = functools.partial(nn.GroupNorm, 8)#functools.partial(InPlaceABNSync, activation='none')
# BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=0, bias=True))


def conv3x3s(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Sequential(nn.ReplicationPad2d(1), torch.nn.utils.spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=0, bias=True)))


def make_blocks_dict(gbuffer_norm, num_gbuffer_layers):
	return {
		'BASIC': (lambda *args, **kwargs: BasicBlock(*args, **{'norm_func':ge.gbuffer_norm_factory(gbuffer_norm, num_gbuffer_layers), **kwargs}), BasicBlock.expansion),
		'BOTTLENECK': (lambda *args, **kwargs: Bottleneck(*args, **{'norm_func':ge.gbuffer_norm_factory(gbuffer_norm, num_gbuffer_layers), **kwargs}), Bottleneck.expansion)
	}



class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, norm_func=ge.gbuffer_norm_factory('Default', 0)):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_func(planes)#, momentum=BN_MOMENTUM)
		self.relu = nn.ReLU(inplace=False)
		self.conv2 = nn.Sequential(nn.ReLU(inplace=True), conv3x3(planes, planes))
		# self.bn2 = norm_func(planes)#, momentum=BN_MOMENTUM)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		x,g = x

		r = x if self.downsample is None else self.downsample(x)

		x = self.conv1(x)
		x = self.bn1(x, g)
		x = self.conv2(x)
		# x = self.bn2(x, g)

		x = 0.1 * x + r
		# x = self.relu(x)

		return [x, g]


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None, norm_func=ge.gbuffer_norm_factory('Default', 0)):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = norm_func(planes)#, momentum=BN_MOMENTUM)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn2 = norm_func(planes)#, momentum=BN_MOMENTUM)
		self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
							   bias=False)
		self.bn3 = norm_func(planes * self.expansion)#,
							   #momentum=BN_MOMENTUM)
		self.relu = nn.ReLU(inplace=False)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out = out + residual
		out = self.relu(out)

		return out


class HighResolutionModule(nn.Module):
	def __init__(self, num_branches, blocks, block_expansion, num_blocks, num_inchannels,
				 num_channels, fuse_method, norm, multi_scale_output=True):
		super(HighResolutionModule, self).__init__()
		self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)

		self.num_inchannels     = num_inchannels
		self.fuse_method        = fuse_method
		self.num_branches       = num_branches
		self.Norm2d             = norm
		self.multi_scale_output = multi_scale_output

		self.branches    = self._make_branches(num_branches, blocks, block_expansion, num_blocks, num_channels)
		self.fuse_layers = self._make_fuse_layers()
		self.relu        = nn.ReLU(inplace=False)

		assert len(self.branches) == num_branches, f'HRModule has {len(self.branches)} branches, but is supposed to have {num_branches}.'
		pass


	def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
		if num_branches != len(num_blocks):
			error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
				num_branches, len(num_blocks))
			logger.error(error_msg)
			raise ValueError(error_msg)

		if num_branches != len(num_channels):
			error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
				num_branches, len(num_channels))
			logger.error(error_msg)
			raise ValueError(error_msg)

		if num_branches != len(num_inchannels):
			error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
				num_branches, len(num_inchannels))
			logger.error(error_msg)
			raise ValueError(error_msg)


	def _make_one_branch(self, branch_index, block, block_expansion, num_blocks, num_channels, stride=1):
		downsample = None
		if stride != 1 or \
		   self.num_inchannels[branch_index] != num_channels[branch_index] * block_expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.num_inchannels[branch_index],
						  num_channels[branch_index] * block_expansion,
						  kernel_size=1, stride=stride, bias=True),
				# self.Norm2d(num_channels[branch_index] * block_expansion)#,
						  #  momentum=BN_MOMENTUM),
			)

		layers = []
		layers.append(block(self.num_inchannels[branch_index],
							num_channels[branch_index], stride, downsample))
		self.num_inchannels[branch_index] = \
			num_channels[branch_index] * block_expansion
		for i in range(1, num_blocks[branch_index]):
			layers.append(block(self.num_inchannels[branch_index],
								num_channels[branch_index]))

		return nn.Sequential(*layers)


	def _make_branches(self, num_branches, block, block_expansion, num_blocks, num_channels):
		branches = []

		for i in range(num_branches):
			branches.append(
				self._make_one_branch(i, block, block_expansion, num_blocks, num_channels, 1))

		return nn.ModuleList(branches)


	def _make_fuse_layers(self):
		if self.num_branches == 1:
			return None

		num_branches = self.num_branches
		num_inchannels = self.num_inchannels
		fuse_layers = []
		for i in range(num_branches if self.multi_scale_output else 1):
			fuse_layer = []
			for j in range(num_branches):
				if j > i:
					fuse_layer.append(nn.Sequential(
						nn.Conv2d(num_inchannels[j],
								  num_inchannels[i],
								  1,
								  1,
								  0,
								  bias=True),
						# self.Norm2d(num_inchannels[i])
						))#, momentum=BN_MOMENTUM)))
				elif j == i:
					fuse_layer.append(None)
				else:
					conv3x3s = []
					for k in range(i-j):
						if k == i - j - 1:
							num_outchannels_conv3x3 = num_inchannels[i]
							conv3x3s.append(nn.Sequential(
								nn.ReplicationPad2d(1),
								nn.Conv2d(num_inchannels[j],
										  num_outchannels_conv3x3,
										  3, 2, 0, bias=True),
								# self.Norm2d(num_outchannels_conv3x3)
								))#, 
										   # momentum=BN_MOMENTUM)))
						else:
							num_outchannels_conv3x3 = num_inchannels[j]
							conv3x3s.append(nn.Sequential(
								nn.ReplicationPad2d(1),
								nn.Conv2d(num_inchannels[j],
										  num_outchannels_conv3x3,
										  3, 2, 0, bias=True),
								# self.Norm2d(num_outchannels_conv3x3),
										  #  momentum=BN_MOMENTUM),
								nn.ReLU(inplace=False)))
					fuse_layer.append(nn.Sequential(*conv3x3s))
			fuse_layers.append(nn.ModuleList(fuse_layer))

		return nn.ModuleList(fuse_layers)

	def get_num_inchannels(self):
		return self.num_inchannels

	def forward(self, x):
		x,g = x
		if self.num_branches == 1:
			return [self.branches[0]([x[0], g[0]])[0]]

		assert len(x) >= self.num_branches, f'HRModule needs feature input for {self.num_branches}, but only got {len(x)}.'
		assert len(g) >= self.num_branches, f'HRModule needs feature input for {self.num_branches}, but only got {len(g)}.'

		for i in range(self.num_branches):
			x[i], _ = self.branches[i]([x[i], g[i]])

		x_fuse = []
		for i in range(len(self.fuse_layers)):
			y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
			for j in range(1, self.num_branches):
				if i == j:
					y = y + x[j]
				elif j > i:
					width_output = x[i].shape[-1]
					height_output = x[i].shape[-2]
					y = y + F.interpolate(
						self.fuse_layers[i][j](x[j]),
						size=[height_output, width_output],
						mode='bilinear')
				else:
					y = y + self.fuse_layers[i][j](x[j])
			x_fuse.append(self.relu(y))

		return [x_fuse, g]



class GBufferEncoderType(Enum):
	NONE    = 0
	CONCAT  = 1
	SPADE   = 2
	ENCODER = 3
	pass


class HighResolutionNet(nn.Module):

	def __init__(self, hr_config, ie_config):
		extra = hr_config
		super(HighResolutionNet, self).__init__()

		self._log = logging.getLogger('epe.network.HRNet')
		self._log.debug('Intializing network ...')

		# instead of a fixed dictionary with basic and bottleneck blocks
		# as in the original HRNet, we generate it on the fly with 
		# blocks that contain a G-buffer normalization module
		self._gbuffer_encoder_norm = ie_config.get('gbuffer_encoder_norm', 'residual')
		self._gbuffer_norm         = ie_config.get('gbuffer_norm', 'RAD')
		self._gbuffer_norm_layers  = ie_config.get('num_gbuffer_layers', '0')
		self._other_norm           = ie_config.get('other_norm', 'batch')
		self._stem_norm            = ie_config.get('stem_norm', 'batch')
		self._num_classes          = ie_config.get('num_classes', 1)
		self._num_gbuf_channels    = int(ie_config.get('num_gbuffer_channels', 0))
		self._num_stages           = int(ie_config.get('num_stages', 6))
		encoder_type = ie_config.get('encoder_type', 3)
		self._encoder_type = GBufferEncoderType[encoder_type]

		self._log.debug(f'  # stages              : {self._num_stages}')
		self._log.debug(f'  G-Buffer encoder type : {self._encoder_type}')
		self._log.debug(f'  G-Buffer encoder norms: {self._gbuffer_encoder_norm}')
		self._log.debug(f'  G-Buffer norm         : {self._gbuffer_norm}')
		self._log.debug(f'  Transition/Fusion norm: {self._other_norm}')
		self._log.debug(f'  # Classes             : {self._num_classes}')
		self._log.debug(f'  # G-Buffer channels   : {self._num_gbuf_channels}')

		blocks_dict = make_blocks_dict(self._gbuffer_norm, self._gbuffer_norm_layers)

		# normalization layers in transition and fusion layers as well as for downsampling
		# are indpendent of G-buffers. here we only chose from regular normalization layers
		self.Norm2d = nf.norm_factory[self._other_norm]

		if self._encoder_type is GBufferEncoderType.CONCAT:
			dim_in = self._num_gbuf_channels + self._num_classes + 3
		else:
			dim_in = 3

		self.stem = nf.make_conv_layer(\
				[dim_in, 16, 16], 
				1, False, False, None)

		if self._encoder_type is GBufferEncoderType.ENCODER:
			self.gbuffer_encoder = ge.GBufferEncoder(0, \
				self._gbuffer_encoder_norm, self._num_classes, 
				self._num_gbuf_channels, ie_config['cls2gbuf'], self._num_stages)
			pass
		elif self._encoder_type is GBufferEncoderType.COMPLEX:
			self.gbuffer_encoder = make_genet(ie_config)
			pass
		else:
			self.gbuffer_encoder = None
			pass	

		self.stage1_cfg    = extra['STAGE1']
		self._log.debug(f'  Stage 1')
		self._log.debug(f'  {self.stage1_cfg}')

		num_channels       = self.stage1_cfg['NUM_CHANNELS'][0]
		block, block_exp   = blocks_dict[self.stage1_cfg['BLOCK']]
		num_blocks         = self.stage1_cfg['NUM_BLOCKS'][0]
		self.layer1        = self._make_layer(block, block_exp, 16, num_channels, num_blocks, 1)
		pre_stage_channels = [block_exp * num_channels]

		stage_cfgs  = []
		stages      = []
		transitions = []
		for si in range(2, self._num_stages+1):
			stage_cfg = extra[f'STAGE{si}']			
			num_channels       = stage_cfg['NUM_CHANNELS']
			block, block_exp   = blocks_dict[stage_cfg['BLOCK']]
			num_channels       = [num_channels[i] * block_exp for i in range(len(num_channels))]
			transitions.append(self._make_transition_layer(pre_stage_channels, num_channels))
			stage, pre_stage_channels = self._make_stage(block, block_exp, stage_cfg, num_channels)
			stages.append(stage)
			stage_cfgs.append(stage_cfg)
			pass

		self.transitions = nn.ModuleList(transitions)
		self.stages      = nn.ModuleList(stages)
		self.stage_cfgs  = stage_cfgs

		out_channels = pre_stage_channels[::-1]

		last_layers = []
		for i,(ci,co) in enumerate(zip(out_channels[:-1], out_channels[1:])):
			m = [nn.ReplicationPad2d(1), nn.Conv2d(ci+co, co, 3), nn.LeakyReLU(0.2, True)]
			if i == self._num_stages-2:
				m += [nn.ReplicationPad2d(1), nn.Conv2d(co, 3, 3)]
				pass
			last_layers.append(nn.Sequential(*m))
			pass

		self.up_layers = nn.ModuleList(last_layers)
		pass


	def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
		num_branches_cur = len(num_channels_cur_layer)
		num_branches_pre = len(num_channels_pre_layer)

		transition_layers = []
		for i in range(num_branches_cur):
			if i < num_branches_pre:
				if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
					transition_layers.append(nn.Sequential(
						nn.ReplicationPad2d(1),
						nn.Conv2d(num_channels_pre_layer[i],
								  num_channels_cur_layer[i],
								  3,
								  1,
								  0,
								  bias=True),
						# self.Norm2d(
							# num_channels_cur_layer[i]),#, momentum=BN_MOMENTUM),
						nn.ReLU(inplace=False)))
				else:
					transition_layers.append(None)
			else:
				conv3x3s = []
				for j in range(i+1-num_branches_pre):
					inchannels = num_channels_pre_layer[-1]
					outchannels = num_channels_cur_layer[i] \
						if j == i-num_branches_pre else inchannels
					conv3x3s.append(nn.Sequential(
						nn.ReplicationPad2d(1),
						nn.Conv2d(
							inchannels, outchannels, 3, 2, 0, bias=True),
						# self.Norm2d(outchannels),#, momentum=BN_MOMENTUM),
						nn.ReLU(inplace=False)))
				transition_layers.append(nn.Sequential(*conv3x3s))

		return nn.ModuleList(transition_layers)


	def _make_layer(self, block, block_expansion, inplanes, planes, num_blocks, stride):
		downsample = None
		if stride != 1 or inplanes != planes * block_expansion:
			downsample = nn.Sequential(
				nn.Conv2d(inplanes, planes * block_expansion,
						  kernel_size=1, stride=stride, bias=True),
				# self.Norm2d(planes * block_expansion),#, momentum=BN_MOMENTUM),
			)

		layers = []
		layers.append(block(inplanes, planes, stride, downsample))
		inplanes = planes * block_expansion
		for i in range(1, num_blocks):
			layers.append(block(inplanes, planes))

		return nn.Sequential(*layers)


	def _make_stage(self, block, block_expansion, layer_config, num_inchannels, multi_scale_output=True):
		num_modules  = layer_config['NUM_MODULES']
		num_branches = layer_config['NUM_BRANCHES']
		num_blocks   = layer_config['NUM_BLOCKS']
		num_channels = layer_config['NUM_CHANNELS']
		# block = blocks_dict[layer_config['BLOCK']]
		fuse_method  = layer_config['FUSE_METHOD']

		modules = []
		for i in range(num_modules):
			# multi_scale_output is only used last module
			if not multi_scale_output and i == num_modules - 1:
				reset_multi_scale_output = False
			else:
				reset_multi_scale_output = True
			modules.append(
				HighResolutionModule(num_branches,
									  block,
									  block_expansion,
									  num_blocks,
									  num_inchannels,
									  num_channels,
									  fuse_method,
									  self.Norm2d,
									  reset_multi_scale_output)
			)
			num_inchannels = modules[-1].get_num_inchannels()

		return nn.Sequential(*modules), num_inchannels


	def forward(self, epe_batch):

		x = epe_batch.img
		g = epe_batch.gbuffers
		s = epe_batch.gt_labels
		del epe_batch

		_,_,h,w = x.shape

		if self._log.isEnabledFor(logging.DEBUG):
			self._log.debug(f'IENet:forward(i:{x.shape}, g:{g.shape}, s:{s.shape})')
			pass

		if self._encoder_type is GBufferEncoderType.CONCAT:
			x = torch.cat((x, g), 1)
			g_list = [None for i in range(4)]
		elif self._encoder_type is GBufferEncoderType.SPADE:
			g_list = [g]
		elif self._encoder_type in [GBufferEncoderType.ENCODER]:
			g_list = self.gbuffer_encoder(g,s)
		elif self._encoder_type in [GBufferEncoderType.COMPLEX]:
			g_list = self.gbuffer_encoder([g,s])
		else:
			g_list = [None for i in range(4)]

		del g
		del s

		if self._log.isEnabledFor(logging.DEBUG) and self._encoder_type not in [GBufferEncoderType.CONCAT]:
			self._log.debug(f'  Encoded G-buffers for {len(g_list)} branches:')
			for i,gi in enumerate(g_list):
				self._log.debug(f'  {i}: {gi.shape}')
				pass				
			pass

		x   = self.stem(x)
		x,_ = self.layer1([x, g_list[0]])


		x_list = [x if self.transitions[0][i] is None else self.transitions[0][i](x) \
			for i in range(self.stage_cfgs[0]['NUM_BRANCHES'])]


		for j in range(self._num_stages-2):
			if self._encoder_type is GBufferEncoderType.SPADE:
				g_list = ge._append_downsampled_gbuffers(g_list, x_list)
				pass

			x_list = [x_list, g_list]
			y_list, _ = self.stages[j](x_list)

			x_list = []
			for i in range(self.stage_cfgs[j+1]['NUM_BRANCHES']):
				if self.transitions[j+1][i] is None:
					x_list.append(y_list[i])
				else:
					x_list.append(self.transitions[j+1][i](y_list[i if i < self.stage_cfgs[j]['NUM_BRANCHES'] else -1]))
					pass
				pass
			pass

		if self._encoder_type is GBufferEncoderType.SPADE:
			g_list = ge._append_downsampled_gbuffers(g_list, x_list)
			pass

		x_list = [x_list, g_list]
		x, _ = self.stages[-1](x_list)
		del y_list
		del x_list
		del g_list

		x = x[::-1]
		y = x[0]
		for i,xi in enumerate(x[1:]):
			y = F.interpolate(y, size=(xi.shape[-2], xi.shape[-1]), mode='bilinear', align_corners=False)
			y = self.up_layers[i](torch.cat((y, xi), 1))
			pass
		return y


	def init_weights(self, pretrained='',):
		logger.info('=> init weights from normal distribution')
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.normal_(m.weight, std=0.001)
			elif isinstance(m, nn.GroupNorm):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)


def make_hrnet_config(num_stages):
	hrnet_cfg = {}

	for i in range(1,num_stages+1):
		cfg = {}
		cfg['NUM_MODULES']  = 1
		cfg['NUM_BRANCHES'] = i
		cfg['NUM_BLOCKS']   = [3]*i
		cfg['NUM_CHANNELS'] = [16 * 2**j for j in range(0,i)]
		cfg['BLOCK']        = 'BASIC'
		cfg['FUSE_METHOD']  = 'SUM'
		hrnet_cfg[f'STAGE{i}'] = cfg
		pass
	return hrnet_cfg


def make_ienet2(ie_config):
	hrnet_config = make_hrnet_config(ie_config.get('num_stages', 6))
	encoder_type = ie_config.get('encoder_type', 3)
	encoder_type = GBufferEncoderType[encoder_type]
	return HighResolutionNet(hrnet_config, ie_config)

