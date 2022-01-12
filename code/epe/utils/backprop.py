import logging
import random

import torch

class Controller:
    def __init__(self, device, target=0.6, max_step=1000, name=''):
        self._log = logging.getLogger(f'epe.backprop.{name}')

        assert type(target) is float

        self.target             = target
        self.num_labels_to_wait = 10

        self.running_stat = torch.tensor([0.0, 0.0], device=device)
        self.r_t_stat = 0
        self.step     = 1
        self.max_step = max_step
        
        self.disabled = target < 0
        pass

    @torch.no_grad()
    def tune(self, correct_labels):
        self.running_stat += torch.tensor(\
            (correct_labels.mean().detach(), correct_labels.shape[0]),
            device=correct_labels.device)

        if self.running_stat[1] > self.num_labels_to_wait - 1:
            sum_accuracies, num_predictions = self.running_stat.tolist()
            self.r_t_stat = sum_accuracies / num_predictions
            sign = 1 if self.r_t_stat > self.target else -1
            self.step  = min(self.max_step, max(0, self.step + sign))
            self.running_stat.mul_(0)
            self.num_labels_to_wait = self.step #* 10

            if self._log.isEnabledFor(logging.DEBUG):
                self._log.debug(f'tune: r_stat: {self.r_t_stat:0.2f}, target: {self.target}, s:{sign}, p:{1.0 / (1.0 + self.step)}, step:{self.step}[{self.max_step}], wait:{self.num_labels_to_wait}')
                pass
            pass

        return 1.0 / (1.0 + self.step)


class AdaptiveBackprop:
    def __init__(self, num_discs, device, target=0.6):
        self.p = [1.01] * num_discs
        self._controllers = [Controller(device, target, name=f'c{i}') for i in range(num_discs)]
        pass

    def sample(self):
        return [random.random() < p if not c.disabled else False for p,c in zip(self.p, self._controllers)]

    def update(self, correct_predictions):
        """ Updates controller for every disc with disc's correct predictions. 
        """
        for i, x in correct_predictions.items():
            self.p[i] = self._controllers[i].tune(torch.cat(x, -1))
            pass
        pass
