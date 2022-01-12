import time
import logging
from pathlib import Path

from scipy.io import savemat
import torch

from torch import autograd

from .BaseExperiment import BaseExperiment

class FeedForwardExperiment(BaseExperiment):
	""" Provide default implementations for a simple feedforward network experiment.

	A feedforward network is a simple network taking some input and producing some output.
	It is a single network.
	"""

	actions  = ['train', 'test', 'infer', 'importance', 'val', 'analyze']
	networks = {}
	
	def __init__(self, args):
		"""Common set up code for all actions."""
		super(FeedForwardExperiment, self).__init__(args)
		pass


	@property
	def i(self):
		return self.state.iterations
	

	def _init_network(self):
		pass


	def _init_dataset(self):
		pass


	def _init_network_state(self):
		""" Initialize optimizer and scheduler for the network. """
		
		o = make_optimizer(self.network.params(), self.args)
		self.state  = NetworkState(self.network, o, make_scheduler(o, args), args)
		pass

	
	def _train_network(self, batch, i):
		self.state.prepare()
		log_scalar, log_img = self._run_network(batch, i)
		self.state.update()

		return log_scalar, log_img


	def _run_network(self, batch, i):
		raise NotImplementedError
		return []


	def evaluate_test(self, batch, batch_id):
		raise NotImplementedError
		pass


	def evaluate_infer(self, sample):
		raise NotImplementedError
		pass


	def _load_sample(self):
		""" Loads a single example (preferably from self.args.input). """
		raise NotImplementedError
		return batch


	def _save_model(self, *, epochs=None, iterations=None, suffix=None):
		
		suffix = f'-{reason}' if reason is not None else ''
		suffix = f'-e{epochs}{suffix}' if epochs is not None else suffix
		suffix = f'-{iterations}{suffix}' if iterations is not None else suffix

		base_filename = self.args.weight_dir / f'{self.args.weight_save}{suffix}'
	
		self.log.info(f'Saving model to {base_filename}.')
		torch.save(self.state.save_to_dict(), f'{base_filename}.pth.tar')		
		pass


	def _load_model(self):
		base_filename = self.args.weight_dir / f'{self.args.weight_init}'
		savegame = torch.load(f'{base_filename}.pth.tar')
		self.state.load_from_dict(savegame)
		pass


	def validate(self,i):
		if len(self.dataset_fake_val) > 0:
			torch.cuda.empty_cache()
			loader_fake = torch.utils.data.DataLoader(self.dataset_fake_val, \
				batch_size=1, shuffle=False, \
				num_workers=self.args.num_loaders, pin_memory=True, drop_last=False, worker_init_fn=seed_worker)

			self.network.eval()

			toggle_grad(self.network.generator, False)
			toggle_grad(self.network.discriminator, False)

			with torch.no_grad():
				for bi, batch_fake in enumerate(loader_fake):
					# last item of batch_fake is just index
					batch_fake = [f.to(self.device, non_blocking=True) for f in batch_fake[:-1]]
					gen_vars = self.forward_generator_fake(batch_fake, i)
					del batch_fake
					self.dump_val(i, bi, gen_vars)
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
			self.log.warning('Validation set is empty - Skipping validation.')
		pass


	


	def dbg(self):
		"""Test a network on a dataset."""
		self.loader_fake = torch.utils.data.DataLoader(self.dataset_fake_val, \
			batch_size=3, shuffle=False, \
			num_workers=4, pin_memory=True, drop_last=False, worker_init_fn=seed_worker)

		if self.args.weight_init:
			self._load_model()
			pass

		self.network.eval()
		with torch.no_grad():
			for bi, batch_fake in enumerate(self.loader_fake):
				batch_fake = [f.to(self.device, non_blocking=True) for f in batch_fake[:-1]]
				_, gen_vars = self.evaluate_dbg(batch_fake)
				self.dump(bi, gen_vars, {}, True)
				pass
			pass
		pass

	def test(self):
		"""Test a network on a dataset."""
		self.loader_fake = torch.utils.data.DataLoader(self.dataset_fake_val, \
			batch_size=1, shuffle=(self.args.shuffle_test), \
			num_workers=self.args.num_loaders, pin_memory=True, drop_last=False, worker_init_fn=seed_worker)

		if self.args.weight_init:
			self._load_model()
			pass

		self.network.eval()
		with torch.no_grad():
			for bi, batch_fake in enumerate(self.loader_fake):                
				print('batch %d' % bi)
				batch_fake = [f.to(self.device, non_blocking=True) for f in batch_fake[:-1]]
				self.save_result(self.evaluate_test(batch_fake, bi), bi)
				pass
			pass
		pass


	def infer(self):
		"""Run network on single example."""

		if self.args.weight_init:
			self._load_model()
			pass

		self.network.train()
		# with torch.no_grad():
		self.evaluate_infer(self._load_sample())
			# pass
		pass

	@classmethod
	def add_arguments(cls, parser):
		# methods available at command line 
		
		super(FeedForwardExperiment, cls).add_arguments(parser)

		parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='[train]')
		parser.add_argument('-m', '--momentum', type=float, default=0.0, help='[train]')
		parser.add_argument('--optim', type=str, choices=['adam', 'sgd', 'adamw'], default='adam')
		parser.add_argument('--adam_ams', action='store_true', default=False)
		parser.add_argument('--adam_beta', type=float, default=0.9)
		parser.add_argument('--adam_beta2', type=float, default=0.999)
		parser.add_argument('--weight_decay', type=float, default=0.0001)
		parser.add_argument('--scheduler', type=str, choices=['step', 'exp', 'cosine'], help='Learning rate scheduler. [train]')
		parser.add_argument('--step', type=int, default=-1)
		parser.add_argument('--step_gamma', type=float, default=-1, help='Step size gamma for learning rate scheduler. [train]')
		return parser


	def run(self):
		self.__getattribute__(self.args.action)()
		pass
	

# if __name__ == '__main__':
	
# 	parser = FeedForwardExperiment.argparser()
# 	args = parser.parse_args()

# 	experiment = FeedForwardExperiment(args)
# 	experiment.run()
