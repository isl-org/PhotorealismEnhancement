import logging

import torch
import torch.nn as nn
from torchvision import models

class GuidedReLUFunc(torch.autograd.Function):	
	@staticmethod
	def forward(ctx, i):
		o = i.clamp(min=0)
		ctx.save_for_backward(o)
		return o

	@staticmethod
	def backward(ctx, grad_output):
		o, = ctx.saved_tensors
		return (o > 0).float() * grad_output.clamp(min=0)

class ReLUWrap(torch.nn.Module):
	def __init__(self, func):
		super(ReLUWrap, self).__init__()
		self.func = func
	
	def forward(self,x):
		return self.func.apply(x)

def norml2(x):
	return x / x.pow(2).sum(dim=1,keepdim=True).sqrt()

class VGG16(torch.nn.Module):
	def __init__(self, requires_grad=False, padding='replicate', replace_reluguided=False):
		super(VGG16, self).__init__()

		self.mean = torch.zeros(1,3,1,1, requires_grad=False)
		self.mean[0,0,0,0] = 0.485
		self.mean[0,1,0,0] = 0.456
		self.mean[0,2,0,0] = 0.406

		self.std = torch.zeros(1,3,1,1, requires_grad=False)
		self.std[0,0,0,0] = 0.229
		self.std[0,1,0,0] = 0.224
		self.std[0,2,0,0] = 0.225

		pretrained_vgg = models.vgg16(pretrained=True)
		features = pretrained_vgg.features
		classifier = pretrained_vgg.classifier

		def convrelu(slice, suffix, conv_id):
			if padding == 'replicate':
				slice.add_module('pad'+suffix, nn.ReplicationPad2d(1))
				features[conv_id].padding = (0,0)
			elif padding == 'zero':
				pass
			elif padding == 'none':
				features[conv_id].padding = (0,0)
				pass

			slice.add_module('conv'+suffix, features[conv_id])
			if replace_reluguided:
				slice.add_module('relu'+suffix, ReLUWrap(GuidedReLUFunc))	
			else:
				slice.add_module('relu'+suffix, nn.ReLU(True))
				pass
			pass

		self.relu_0 = nn.Sequential()
		convrelu(self.relu_0, '1_1', 0)

		self.relu_1 = nn.Sequential()
		convrelu(self.relu_1, '1_2', 2)

		self.relu_2 = nn.Sequential()
		self.relu_2.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
		convrelu(self.relu_2, '2_1', 5)

		self.relu_3 = nn.Sequential()
		convrelu(self.relu_3, '2_2', 7)

		self.relu_4 = nn.Sequential()		
		self.relu_4.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
		convrelu(self.relu_4, '3_1', 10)
		
		self.relu_5 = nn.Sequential()
		convrelu(self.relu_5, '3_2', 12)

		self.relu_6 = nn.Sequential()		
		convrelu(self.relu_6, '3_3', 14)
		
		self.relu_7 = nn.Sequential()
		self.relu_7.add_module('pool3', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
		convrelu(self.relu_7, '4_1', 17)

		self.relu_8 = nn.Sequential()		
		convrelu(self.relu_8, '4_2', 19)

		self.relu_9 = nn.Sequential()		
		convrelu(self.relu_9, '4_3', 21)

		self.relu_10 = nn.Sequential()
		self.relu_10.add_module('pool4', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
		convrelu(self.relu_10, '5_1', 24)

		self.relu_11 = nn.Sequential()		
		convrelu(self.relu_11, '5_2', 26)

		self.relu_12 = nn.Sequential()		
		convrelu(self.relu_12, '5_3', 28)

		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

		self.fc_0 = nn.Sequential()
		self.fc_0.add_module('fc6', classifier[0])

		self.fc_1 = nn.Sequential()
		if replace_reluguided:
			self.fc_1.add_module('relu6', ReLUWrap(GuidedReLUFunc))	
		else:
			self.fc_1.add_module('relu6', nn.ReLU(True))
			pass
		self.fc_1.add_module('dropout', nn.Dropout())
		self.fc_1.add_module('fc7', classifier[3])

		self.fc_2 = nn.Sequential()
		if replace_reluguided:
			self.fc_2.add_module('relu7', ReLUWrap(GuidedReLUFunc))	
		else:
			self.fc_2.add_module('relu7', nn.ReLU(True))
			pass
		self.fc_2.add_module('dropout', nn.Dropout())
		self.fc_2.add_module('fc8', classifier[6])

		if not requires_grad:
			for param in self.parameters():
				param.requires_grad = False
				pass
			pass
		pass

	def to(self, device, **kwargs):
		logging.info(f'VGG16:to: {torch.cuda.memory_allocated(device=device)}')
		self.mean = self.mean.to(device, **kwargs)
		self.std  = self.std.to(device, **kwargs)
		return super().to(device, **kwargs)

	def set_mean_std(self, mr,mg,mb, sr,sg,sb):
		self.mean[0,0,0,0] = mr
		self.mean[0,1,0,0] = mg
		self.mean[0,2,0,0] = mb
		self.std[0,0,0,0] = sr
		self.std[0,1,0,0] = sg
		self.std[0,2,0,0] = sb
		pass

	def normalize(self, x):
		return (x - self.mean) / self.std
		
	def fw_relu(self, x, num_relus, do_normalize=True):
		if do_normalize:
			x = self.normalize(x)
			pass

		out = []
		for i in range(num_relus):
			x = getattr(self, 'relu_%d' % i)(x)
			out.append(x)
			pass
		return out

	def fw_fc(self, x, num_fcs, do_normalize=True):
		if do_normalize:
			x = self.normalize(x)
			pass

		for i in range(13):
			x = getattr(self, 'relu_%d' % i)(x)
			pass

		x = self.avgpool(x)
		x = torch.flatten(x, 1)

		out = []
		for i in range(num_fcs):
			x = getattr(self, 'fc_%d' % i)(x)
			out.append(x)
			pass
		return out		


