import time
import logging
from pathlib import Path

from scipy.io import savemat
import torch
from torch import autograd

from .BaseExperiment import BaseExperiment, NetworkState, toggle_grad, seed_worker

class GANExperiment(BaseExperiment):
	actions  = ['train', 'test', 'infer']
	networks = {}

	def __init__(self, args):
		"""Common set up code for all actions."""
		super(GANExperiment, self).__init__(args)

		pass

	def _parse_config(self):
		super()._parse_config()

		self._gen_cfg  = dict(self.cfg.get('generator', {}))		
		self._disc_cfg = dict(self.cfg.get('discriminator', {}))
		pass


	@property
	def i(self):
		return self.gen_state.iterations + self.disc_state.iterations


	def _init_network(self):
		pass


	def _init_network_state(self):
		""" Initialize optimizer and scheduler for the network. """

		self.gen_state  = NetworkState(self.network.generator, self._gen_cfg, 'generator')
		self.disc_state = NetworkState(self.network.discriminator, self._disc_cfg, 'discriminator')
		pass


	def _init_dataset(self):
		pass


	def _train_network(self, batch):
		if self.i % 2 == 0:
			log_scalar, log_img = self._train_discriminator(batch, self.i)
		else:
			log_scalar, log_img = self._train_generator(batch, self.i)

		return log_scalar, log_img


	def _train_discriminator(self, batch, i):
		""" Execute an optimization step for the discriminator. """

		toggle_grad(self.network.generator, False)
		toggle_grad(self.network.discriminator, True)

		self.disc_state.prepare()
		log_scalar, log_img = self._run_discriminator(batch.fake, batch.real, i)
		self.disc_state.update()
		# self._profiler.step()

		return log_scalar, log_img


	def _train_generator(self, batch, i):
		""" Execute an optimization step for the generator. """

		toggle_grad(self.network.generator, True)
		toggle_grad(self.network.discriminator, False)

		self.gen_state.prepare()
		log_scalar, log_img = self._run_generator(batch.fake, batch.real, i)				
		self.gen_state.update()
		# self._profiler.step()

		return log_scalar, log_img


	def _run_generator(self, batch, batch_id):
		""" Run a forward and backward pass on the generator.

		This function is called within an optimization step for the generator.
		It contains the data and network specific code.
		"""

		raise NotImplementedError
		return []


	def _run_discriminator(self, batch, batch_id):
		""" Run a forward and backward pass on the generator.

		This function is called within an optimization step for the generator.
		It contains the data and network specific code.
		"""

		raise NotImplementedError
		return []


	def evaluate_test(self, batch, batch_id):
		raise NotImplementedError
		pass


	def _save_model(self, *, epochs=None, iterations=None, reason=None):

		suffix = f'-{reason}' if reason is not None else ''
		suffix = f'-e{epochs}{suffix}' if epochs is not None else suffix
		suffix = f'-{iterations}{suffix}' if iterations is not None else suffix

		base_filename = self.weight_dir / f'{self.weight_save}{suffix}'
		self._log.info(f'Saving model to {base_filename}.')

		sd, od = self.gen_state.save_to_dict()
		for k,v in sd.items():
			try:
				torch.save(v, f'{base_filename}_gen-{k}.pth.tar')
			except:
				self._log.error('Cannot store {k}.')

		sd, od = self.disc_state.save_to_dict()
		for k,v in sd.items():
			torch.save(v, f'{base_filename}_disc-{k}.pth.tar')
			pass
		pass


	def _load_model(self):
		""" Load a generator and a discriminator with networks states each from file. """
		
		base_filename = self.weight_dir / f'{self.weight_init}'

		savegame = {}
		for k in ['network', 'optimizer', 'scheduler']:		
			savegame[k]	= torch.load(f'{base_filename}_gen-{k}.pth.tar')
			pass
		self.gen_state.load_from_dict(savegame)

		# discriminator only for training
		if self.action == 'train':
			savegame = {}		
			for k in ['network', 'optimizer', 'scheduler']:		
				savegame[k]	= torch.load(f'{base_filename}_disc-{k}.pth.tar')
				pass
			self.disc_state.load_from_dict(savegame)
			pass
		pass


	def validate(self):
		if not self.no_validation and len(self.dataset_fake_val) > 0:

			torch.cuda.empty_cache()
			loader_fake = torch.utils.data.DataLoader(self.dataset_fake_val, \
				batch_size=1, shuffle=False, \
				num_workers=self.num_loaders, pin_memory=True, drop_last=False, collate_fn=self.collate_fn_val, worker_init_fn=seed_worker)

			self.network.eval()

			toggle_grad(self.network.generator, False)
			toggle_grad(self.network.discriminator, False)

			with torch.no_grad():
				for bi, batch_fake in enumerate(loader_fake):
					
					gen_vars = self._forward_generator_fake(batch_fake.to(self.device))
					del batch_fake
					
					self.dump_val(self.i, bi, gen_vars)
					del gen_vars
					pass
				pass

			self.network.train()

			toggle_grad(self.network.generator, False)
			toggle_grad(self.network.discriminator, True)

			del loader_fake			
			#del gen_vars
			torch.cuda.empty_cache()
			pass
		else:
			self._log.warning('Validation set is empty - Skipping validation.')
		pass


	def test(self):
		"""Test a network on a dataset."""
		self.loader_fake = torch.utils.data.DataLoader(self.dataset_fake_val, \
			batch_size=1, shuffle=(self.shuffle_test), \
			num_workers=self.num_loaders, pin_memory=True, drop_last=False, collate_fn=self.collate_fn_val, worker_init_fn=seed_worker)

		if self.weight_init:
			self._load_model()
			pass

		self.network.eval()

		with torch.no_grad():
			for bi, batch_fake in enumerate(self.loader_fake):                
				print('batch %d' % bi)
				self.save_result(self.evaluate_test(batch_fake.to(self.device), bi), bi)
				pass
			pass
		pass

