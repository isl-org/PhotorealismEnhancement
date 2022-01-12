import torch
import torch.nn as nn

@torch.jit.script
def make_residual(img, x):
	return torch.sigmoid(-torch.log(1 / img.clamp(min=0.001, max=0.999) - 1) + x)


class ResidualGenerator(nn.Module):
	def __init__(self, network):
		super(ResidualGenerator, self).__init__()
		self.network = network
		pass

	def forward(self, epe_batch):
		return make_residual(epe_batch.img, self.network(epe_batch))

