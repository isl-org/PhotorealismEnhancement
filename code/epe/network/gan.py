import torch.nn as nn

class GAN(nn.Module):
	def __init__(self, generator, discriminator):
		super(GAN, self).__init__()
		self.discriminator = discriminator
		self.generator     = generator
		pass
