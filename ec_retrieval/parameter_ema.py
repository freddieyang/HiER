"""A separate exponential parameter average; the online optimizer is untouched.

EMA uses a declared epoch half-life, converted using the planned TRAIN steps.
All parameters are averaged, including the EC router and score scalars. Buffers
are copied exactly; BatchNorm is rejected because it would require a separate
TRAIN-only running-statistics protocol. The exported model is an ordinary HiER.
"""
from copy import deepcopy
import math

import torch
from torch import nn


class ParameterEMA:
    def __init__(self, model, *, half_life_epochs, steps_per_epoch, warmup_epochs):
        if not math.isfinite(half_life_epochs) or half_life_epochs <= 0:
            raise ValueError('A positive finite EMA half-life is required')
        if steps_per_epoch < 1 or int(steps_per_epoch) != steps_per_epoch or warmup_epochs < 0:
            raise ValueError('Positive integral epoch steps and nonnegative warmup required')
        if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()):
            raise ValueError('This controlled EMA does not support BatchNorm statistics')
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('The controlled optimizer and EMA require FP32 parameters')
        self.model = deepcopy(model).eval().requires_grad_(False)
        self.decay = math.exp(-math.log(2.) / (half_life_epochs * steps_per_epoch))
        self.half_life_epochs = float(half_life_epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.warmup_steps = int(warmup_epochs * steps_per_epoch)
        self.updates = 0
        self._names = tuple(dict(model.named_parameters()))
        self._buffer_names = tuple(dict(model.named_buffers()))

    @torch.no_grad()
    def update(self, online):
        if tuple(dict(online.named_parameters())) != self._names:
            raise ValueError('Online parameter identities changed')
        if tuple(dict(online.named_buffers())) != self._buffer_names:
            raise ValueError('Online buffer identities changed')
        target, source = list(self.model.parameters()), list(online.parameters())
        # Copy through the fixed auxiliary-loss warmup. The first post-warmup
        # step averages with the preceding online iterate, never random weights.
        if self.updates < self.warmup_steps or self.updates == 0:
            torch._foreach_copy_(target, source)
        else:
            torch._foreach_lerp_(target, source, 1. - self.decay)
        for averaged, current in zip(self.model.buffers(), online.buffers()):
            averaged.copy_(current)
        self.updates += 1

    def state_dict(self):
        return dict(model_state=self.model.state_dict(), updates=self.updates,
                    decay=self.decay, half_life_epochs=self.half_life_epochs,
                    steps_per_epoch=self.steps_per_epoch, warmup_steps=self.warmup_steps,
                    inference_ensemble_members=1, training_source='TRAIN online updates only')
