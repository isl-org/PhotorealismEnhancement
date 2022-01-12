import os
from struct import unpack

import numpy as np

import torch
import scipy.io as sio

k_save = 0
def save(c, d, name=None):
	global k_save
	if c:
		k_save += 1
		if name==None:
			name = 'out_%d.mat' % k_save
		sio.savemat(name, {k:d[k].detach().cpu().numpy() for k in d.keys()})

def checknan(a, name, d=None):
	if torch.any(torch.isnan(a)):
		print('%s is nan.' % name)
		if d is None:
			save(True, {name:a})
		else:
			save(True, d)
		exit()
		
def mat2tensor(mat):    
	t = torch.from_numpy(mat).float()
	if mat.ndim == 2:
		return t.unsqueeze(2).permute(2,0,1)
	elif mat.ndim == 3:
		return t.permute(2,0,1)


def normalize_dim(a, d):
	""" Normalize a along dimension d."""
	return a.mul(a.pow(2).sum(dim=d,keepdim=True).clamp(min=0.00001).rsqrt())


def cross3(a,b):
	c = a.new_zeros(a.shape[0],3)
	c[:,0] = a[:,1].mul(b[:,2]) - a[:,2].mul(b[:,1])
	c[:,1] = a[:,2].mul(b[:,0]) - a[:,0].mul(b[:,2])
	c[:,2] = a[:,0].mul(b[:,1]) - a[:,1].mul(b[:,0])
	return c


def normalize_vec(a):
	# assert a.shape[-1] == 3 || a.sh
	return a.div(a.pow(2).sum(dim=-1,keepdim=True).sqrt())
