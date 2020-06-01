import numpy as np

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr, n_warmup_steps=10000, warmup=0.3):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = lr
        self.warmup = warmup

    def step_and_update_lr(self, progress=0.5):
        "Step with the inner optimizer"
        l = self._update_learning_rate(progress)
        self._optimizer.step()
        return l

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])
    


    def _update_learning_rate(self, progress=0.5):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self.linear_lr_scal(progress)

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def linear_lr_scal(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return max((progress - 1.) / (self.warmup - 1.), 0.)