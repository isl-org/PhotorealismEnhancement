import logging

import lpips
import torch
import torch.nn as nn


class LPIPSLoss(nn.Module):
	def __init__(self, net):
		super(LPIPSLoss, self).__init__()
		self.model = lpips.LPIPS(lpips=True, net=net, spatial=False, verbose=False)
		for param in self.parameters():
			param.requires_grad = False
			pass
		pass

	def forward_fake(self, img, rec):
		return self.model.forward(img, rec, retPerLayer=False, normalize=True)[0], []
		

def vgg_munit(vgg, img, rec):
		
	ff = torch.nn.functional.instance_norm(vgg.fw_relu(img, 13)[-1])
	fn = torch.nn.functional.instance_norm(vgg.fw_relu(rec, 13)[-1])

	vgg_imgs = []
	vgg_imgs.append((ff-fn).pow(2).mean(dim=1,keepdim=True))
	loss = vgg_imgs[-1].mean()
	
	return loss, vgg_imgs		


def vgg_johnson(vgg, img, rec):
		
	ff = vgg.fw_relu(img, 4)[-1]
	fn = vgg.fw_relu(rec, 4)[-1]

	vgg_imgs = []
	vgg_imgs.append((ff-fn).pow(2).mean(dim=1,keepdim=True))
	loss = vgg_imgs[-1].mean()
	
	return loss, vgg_imgs		


loss_funcs = {'munit':vgg_munit, 'johnson':vgg_johnson}


class VGGLoss(nn.Module):
	def __init__(self, vgg, loss):
		super(VGGLoss, self).__init__()
		self.vgg = vgg
		self.loss_func = loss_funcs[loss]
		pass
	
	def forward_fake(self, img, rec):
		return self.loss_func(self.vgg, img, rec)
