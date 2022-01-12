import warnings
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from argparse import ArgumentParser
import datetime
import logging
from pathlib import Path
import random

import imageio
import numpy as np
from skimage.transform import resize
import torch
import torch.utils.data
from torch import autograd

import kornia as K

import epe.utils
import epe.dataset as ds
import epe.network as nw
import epe.experiment as ee
from epe.matching import MatchedCrops, IndependentCrops

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

logger = logging.getLogger('main')


gan_losses = {\
	'ls':nw.LSLoss,
	'ns':nw.NSLoss,
	'hinge':nw.HingeLoss}


vgg_losses = {\
	'lpips_alex':    lambda vgg: nw.LPIPSLoss(net='alex'),
	'lpips_squeeze': lambda vgg: nw.LPIPSLoss(net='squeeze'),
	'lpips_vgg':     lambda vgg: nw.LPIPSLoss(net='vgg'),
	'munit':         lambda vgg: nw.VGGLoss(vgg, 'munit'),
	'johnson':       lambda vgg: nw.VGGLoss(vgg, 'johnson'),
}


fake_datasets = {\
	'GTA':ds.PfDDataset,
}


dataset_pairings = [\
	'matching',
	'matchinghd',
	'independent_256',
	'independent_256',
	'independent_400',
]

def tee_loss(x, y):
	return x+y, y.detach()


def accuracy(pred):
	return (pred > 0.5).float().mean()


def real_penalty(loss, real_img):
	''' Compute penalty on real images. '''
	b = real_img.shape[0]
	grad_out = autograd.grad(outputs=loss, inputs=[real_img], create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)
	logger.debug(f'real_penalty: g:{grad_out[0].shape}')
	reg_loss = torch.cat([g.pow(2).reshape(b, -1).sum(dim=1, keepdim=True) for g in grad_out if g is not None], 1).mean()
	return reg_loss


class PassthruGenerator(torch.nn.Module):
	def __init__(self, network):
		super(PassthruGenerator, self).__init__()
		self.network = network
		pass

	def forward(self, epe_batch):
		return self.network(epe_batch)


class EPEExperiment(ee.GANExperiment):
	def __init__(self, args):
		super(EPEExperiment, self).__init__(args)
		self.collate_fn_train = ds.JointEPEBatch.collate_fn
		self.collate_fn_val   = ds.EPEBatch.collate_fn
		pass


	def _parse_config(self):
		super()._parse_config()

		# fake dataset

		fake_cfg = dict(self.cfg.get('fake_dataset', {}))
		self.fake_name       = str(fake_cfg.get('name'))
		self.fake_train_path = Path(fake_cfg.get('train_filelist', None))
		self.fake_val_path   = Path(fake_cfg.get('val_filelist', None))
		self.fake_test_path  = Path(fake_cfg.get('test_filelist', None))

		self._log.debug(f'  Fake dataset {self.fake_name} in {self.fake_train_path}[train], {self.fake_val_path}[val], {self.fake_test_path}[test].')

		# real dataset

		real_cfg = dict(self.cfg.get('real_dataset', {}))
		self.real_name     = str(real_cfg.get('name'))
		self.real_basepath = Path(real_cfg.get('filelist', None))

		self._log.debug(f'  Real dataset {self.real_name} in {self.real_basepath}.')

		# sampling 

		self.sample_cfg = dict(fake_cfg.get('sampling', {}))
		self.sampling      = str(self.sample_cfg.get('type', 'matching'))

		# networks

		self.vgg = nw.VGG16().to(self.device)

		gen_cfg = dict(self.cfg.get('generator', {}))
		self.gen_cfg = dict(gen_cfg.get('config', {}))

		disc_cfg = dict(self.cfg.get('discriminator', {}))
		self.disc_cfg = dict(disc_cfg.get('config', {}))

		# loss functions

		loss_cfg = dict(self.cfg.get('objectives', {}))
		self.gan_loss = gan_losses[str(loss_cfg.get('gan', 'ls'))]().to(self.device)

		perc_cfg = dict(loss_cfg.get('perceptual', {}))
		vgg_type = str(perc_cfg.get('type', 'lpips_vgg'))
		self.vgg_loss   = vgg_losses[vgg_type](self.vgg).to(self.device)
		self.vgg_weight = float(perc_cfg.get('weight', 1.0))

		reg_cfg = dict(loss_cfg.get('reg', {}))
		self.reg_weight = float(reg_cfg.get('weight', 1.0))
		pass


	def _init_dataset(self):

		self._log.debug('Initializing datasets ...')

		# validation
		if self.no_validation:
			self.dataset_fake_val = None
		elif self.action == 'test':
			self.dataset_fake_val = fake_datasets[self.fake_name](ds.utils.read_filelist(self.fake_test_path, 4, True))
		else:
			self.dataset_fake_val = fake_datasets[self.fake_name](ds.utils.read_filelist(self.fake_val_path, 4, True))
			pass

		# training

		if self.action == 'train':

			source_dataset = fake_datasets[self.fake_name](ds.utils.read_filelist(self.fake_train_path, 4, True))
			target_dataset = ds.RobustlyLabeledDataset(self.real_name, ds.utils.read_filelist(self.real_basepath, 2, True))

			if self.sampling == 'matching':
				self.dataset_train = MatchedCrops(source_dataset, target_dataset, self.sample_cfg)
			elif self.sampling.startswith('independent_'):
				crop_size = int(self.sampling[len('independent_'):])
				self.dataset_train = IndependentCrops(source_dataset, target_dataset, self.sample_cfg)
			else:
				raise NotImplementedError
			pass
		else:
			self.dataset_train = None
		pass


	def _init_network(self):

		self._log.debug('Initializing networks ...')

		# network arch depends on dataset
		if self.dataset_train is not None:
			self.gen_cfg['num_classes']          = self.dataset_train.source.num_classes
			self.gen_cfg['num_gbuffer_channels'] = self.dataset_train.source.num_gbuffer_channels
			self.gen_cfg['cls2gbuf']             = self.dataset_train.source.cls2gbuf
		else:
			self.gen_cfg['num_classes']          = self.dataset_fake_val.num_classes
			self.gen_cfg['num_gbuffer_channels'] = self.dataset_fake_val.num_gbuffer_channels
			self.gen_cfg['cls2gbuf']             = self.dataset_fake_val.cls2gbuf

		self._log.debug(f'Fake dataset has {self.gen_cfg["num_classes"]} classes and {self.gen_cfg["num_gbuffer_channels"]} G-buffers.')
		self._log.debug(f'Classes are mapped to G-Buffers via {self.gen_cfg["cls2gbuf"]}.')

		generator_type     = self.cfg['generator']['type']
		discriminator_type = self.cfg['discriminator']['type']
		run_disc_always    = bool(self.cfg['discriminator'].get('run_always', False))
		self.check_fake_for_backprop = bool(self.cfg['discriminator'].get('check_fake_for_backprop', True))
		backprop_target = float(self.cfg['discriminator'].get('backprop_target', 0.6))


		if generator_type == 'hr':
			generator = nw.ResidualGenerator(nw.make_ienet(self.gen_cfg))
			pass
		elif generator_type == 'hr_new':
			generator = PassthruGenerator(nw.make_ienet2(self.gen_cfg))
			pass

		discriminator = {\
			'patchgan':nw.PatchGANDiscriminator,
			'pde':nw.PerceptualDiscEnsemble,
			'ppde':nw.PerceptualProjectionDiscEnsemble,
		}[discriminator_type](self.disc_cfg)

		self.network           = nw.GAN(generator, discriminator).to(self.device)
		self.adaptive_backprop = epe.utils.AdaptiveBackprop(len(self.network.discriminator), self.device, backprop_target) if not run_disc_always else None
		self._log.debug(f'AdaptiveBackprop is [{"on" if self.adaptive_backprop else "off"}].')
		self._log.debug(f'  check fake performance : [{"on" if self.check_fake_for_backprop else "off"}].')
		self._log.debug(f'  target                 : {backprop_target}')

		self._log.debug('Networks are initialized.')
		self._log.info(f'{self.network}')
		pass


	def _run_generator(self, batch_fake, batch_real, batch_id):

		rec_fake     = self.network.generator(batch_fake)

		realism_maps = self.network.discriminator.forward(\
			vgg=self.vgg, img=rec_fake, robust_labels=batch_fake.robust_labels, 
			fix_input=False, run_discs=True)

		loss     = 0
		log_info = {}
		for i, rm in enumerate(realism_maps):
			loss, log_info[f'gs{i}'] = tee_loss(loss, self.gan_loss.forward_gen(rm[0,:,:,:].unsqueeze(0)).mean())
			pass

		loss, log_info['vgg'] = tee_loss(loss, self.vgg_weight * self.vgg_loss.forward_fake(batch_fake.img, rec_fake)[0])
		loss.backward()

		return log_info, \
		{'rec_fake':rec_fake.detach(), 'fake':batch_fake.img.detach(), 'real':batch_real.img.detach()}


	def _forward_generator_fake(self, batch_fake):
		""" Run the generator without any loss computation. """

		rec_fake = self.network.generator(batch_fake)
		return {'rec_fake':rec_fake.detach(), 'fake':batch_fake.img.detach()}


	def _run_discriminator(self, batch_fake, batch_real, batch_id:int):

		log_scalar = {}
		log_img    = {}

		# sample probability of running certain discriminator
		if self.adaptive_backprop is not None:
			run_discs = self.adaptive_backprop.sample()
		else:
			run_discs = [True] * len(self.network.discriminator)
			pass

		if not any(run_discs):
			return log_scalar, log_img

		with torch.no_grad():
			rep_fake = self.network.generator(batch_fake)
			pass

		log_img['fake']     = batch_fake.img.detach()
		log_img['rec_fake'] = rep_fake.detach()

		rec_fake = rep_fake.detach()
		rec_fake.requires_grad_()

		# forward fake images
		realism_maps = self.network.discriminator.forward(\
			vgg=self.vgg, img=rec_fake, robust_labels=batch_fake.robust_labels, 
			fix_input=True, run_discs=run_discs)

		loss = 0
		pred_labels = {} # for adaptive backprop
		for i, rm in enumerate(realism_maps):
			if rm is None:
				continue

			if self._log.isEnabledFor(logging.DEBUG):
				log_img[f'realism_fake_{i}'] = rm.detach()
				pass

			# for getting probability of back
			if self.check_fake_for_backprop:
				pred_labels[i] = [(rm.detach() < 0.5).float().reshape(1,-1)]
				pass
			log_scalar[f'rdf{i}']      = accuracy(rm.detach()) # percentage of fake predicted as real
			loss, log_scalar[f'ds{i}'] = tee_loss(loss, self.gan_loss.forward_fake(rm).mean())
			pass
		del rm
		del realism_maps

		loss.backward()


		log_img['real'] = batch_real.img.detach()	
		batch_real.img.requires_grad_()

		# forward real images
		realism_maps = self.network.discriminator.forward(\
			vgg=self.vgg, img=batch_real.img, robust_labels=batch_real.robust_labels, robust_img=batch_real.img, 
			fix_input=(self.reg_weight <= 0), run_discs=run_discs)

		loss = 0		
		for i, rm in enumerate(realism_maps):
			if rm is None:
				continue

			if self._log.isEnabledFor(logging.DEBUG):
				log_img[f'realism_real_{i}'] = rm.detach()
				pass

			if i in pred_labels:
				# predicted correctly, here real as real
				pred_labels[i].append((rm.detach() > 0.5).float().reshape(1,-1))
			else:
				pred_labels[i] = [(rm.detach() > 0.5).float().reshape(1,-1)]
				pass

			log_scalar[f'rdr{i}'] = accuracy(rm.detach()) # percentage of real predicted as real	
			loss += self.gan_loss.forward_real(rm).mean()
			pass
		del rm
		del realism_maps

		# compute gradient penalty on real images
		if self.reg_weight > 0:			
			loss.backward(retain_graph=True)
			self._log.debug(f'Computing penalty on real: {loss} from i:{batch_real.img.shape}.')
			reg_loss, log_scalar['reg'] = tee_loss(0, real_penalty(loss, batch_real.img))
			(self.reg_weight * reg_loss).backward()
		else:
			loss.backward()
			pass
		pass

		# update disc probabilities
		if self.adaptive_backprop is not None:
			self.adaptive_backprop.update(pred_labels)
			pass

		return log_scalar, log_img


	def evaluate_test(self, batch_fake, batch_id):
		new_img = self.network.generator(batch_fake)
		return new_img, batch_fake.img, batch_fake.path[0].stem


	def results_exists(self, id):
		return (self.args.dbg_dir / self.args.weight_save / img_name).exists()


	def save_result(self, results, id):
		new_img, old_img, filename = results

		# img_name = self.dataset_fake_val.fakes[id]
		# if type(img_name) is tuple:
		# 	img_name = [str(t) for t in img_name]
		# 	img_name = '_'.join(img_name)+'.png'
		# 	pass
		img = (new_img[0,...].clamp(min=0,max=1).permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8)
		imageio.imwrite(str(self.dbg_dir / self.weight_save / f'{filename}{self.result_ext}'), img[:,:,:3])
		pass
	pass


if __name__ == '__main__':

	parser = ArgumentParser()
	EPEExperiment.add_arguments(parser)
	args = parser.parse_args()

	ee.init_logging(args)

	experiment = EPEExperiment(args)
	experiment.run()
