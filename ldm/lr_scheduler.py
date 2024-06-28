import numpy as np
import math

class LambdaWarmUpCosineScheduler:
    """
    note: use with a base_lr of 1.0
    """

    def __init__(self, warm_up_steps, lr_min, lr_max, lr_start, max_decay_steps, verbosity_interval=0):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.0
        self.verbosity_interval = verbosity_interval

    def schedule(self, n, **kwargs):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        else:
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(t * np.pi))
            self.last_lr = lr
            return lr

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)


class LambdaWarmUpCosineScheduler2:
    """
    supports repeated iterations, configurable via lists
    note: use with a base_lr of 1.0.
    """

    def __init__(self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.0
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n):
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"current step: {n}, recent lr-multiplier: {self.last_f}, " f"current cycle {cycle}")
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            t = (n - self.lr_warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
            t = min(t, 1.0)
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (1 + np.cos(t * np.pi))
            self.last_f = f
            return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)


class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):
    '''
    Example usage:
    target: ldm.lr_scheduler.LambdaLinearScheduler
      params:
        warm_up_steps: [ 1000 ]
        cycle_lengths: [ 10000000000000 ] # incredibly large number to prevent corner cases
        f_start: [ 1.e-6 ]
        f_max: [ 1. ]
        f_min: [ 1. ]


    '''


    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"current step: {n}, recent lr-multiplier: {self.last_f}, " f"current cycle {cycle}")

        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (self.cycle_lengths[cycle] - n) / (
                self.cycle_lengths[cycle]
            )
            self.last_f = f
            return f

class CosineScheduler:
    """
    Example usage:

    target: ldm.lr_scheduler.CosineScheduler
        params:
            lr_warmup_steps: 100
            lr_max: 1
            lr_start: 1e-3
            total_training_steps: 18000 # This should be dataloader size * epochs / batch_size.
            CosineScheduler(lr_warmup_steps=100, lr_max=0.1, lr_start=1e-2, total_training_steps=total_steps)

    """

    def __init__(self, lr_warmup_steps, lr_max, lr_start, total_training_steps):

        self.lr_max = lr_max
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_start = lr_start
        self.total_training_steps = total_training_steps
        self.last_f = 0.0
        self.lr_increment = (self.lr_max - self.lr_start) / self.lr_warmup_steps
    # total_training_steps = len(train_loader) * n_epoch
    # dataset-17-bravo-1 has length 13553.
    # There are 12 workers, does this also impact it? idk.

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)

    def schedule(self, n, **kwargs):
        if n < self.lr_warmup_steps:
            lr = self.lr_start + n * self.lr_increment
        else:
            #Implement cos decay:
            progress = ((n - self.lr_warmup_steps) /
                        (self.total_training_steps - self.lr_warmup_steps))
            lr = self.lr_start + (self.lr_max - self.lr_start) * 0.5 * (1 + math.cos(math.pi * progress))
        return lr
