from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from tensor import Tensor


class Optimizer:
    def __init__(self, params: Iterable[Tensor]):
        self.params = [param for param in params if param.requires_grad]

    def step(self):
        raise NotImplementedError

    def reset_grad(self):
        for param in self.params:
            param.grad = None

    zero_grad = reset_grad

    def clip_grad_norm(self, max_norm=0.25):
        grads = [param.grad for param in self.params if param.grad is not None]
        if not grads:
            return 0.0

        total_norm = float(
            np.sqrt(
                sum(
                    np.sum(np.square(grad.detach().numpy(), dtype=np.float64))
                    for grad in grads
                )
            )
        )
        scale = min(max_norm / (total_norm + 1e-6), 1.0)
        for param in self.params:
            if param.grad is not None:
                param.grad = param.grad.detach() * scale
        return total_norm


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad.detach()
            if self.weight_decay:
                grad = grad + param.detach() * self.weight_decay

            update = grad
            if self.momentum:
                update = self.momentum * self.velocity.get(param, 0) + grad
                self.velocity[param] = update.data

            param.cached_data = (
                param.detach() - self.lr * update
            ).realize_cached_data()


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad.detach()
            if self.weight_decay:
                grad = grad + param.detach() * self.weight_decay

            m = self.beta1 * self.m.get(param, 0) + (1 - self.beta1) * grad
            v = self.beta2 * self.v.get(param, 0) + (1 - self.beta2) * (grad**2)
            self.m[param] = m.data
            self.v[param] = v.data

            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            update = m_hat / (v_hat**0.5 + self.eps)
            param.cached_data = (
                param.detach() - self.lr * update
            ).realize_cached_data()


class AdamW(Adam):
    def step(self):
        self.t += 1
        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad.detach()
            m = self.beta1 * self.m.get(param, 0) + (1 - self.beta1) * grad
            v = self.beta2 * self.v.get(param, 0) + (1 - self.beta2) * (grad**2)
            self.m[param] = m.data
            self.v[param] = v.data

            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            update = m_hat / (v_hat**0.5 + self.eps)
            if self.weight_decay:
                update = update + param.detach() * self.weight_decay
            param.cached_data = (
                param.detach() - self.lr * update
            ).realize_cached_data()


__all__ = ["Adam", "AdamW", "Optimizer", "SGD"]
