from __future__ import annotations

import numpy as np

import nn

from .data import get_batch


def _detach_state(state):
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(_detach_state(item) for item in state)
    return state.detach()


def epoch(
    batches,
    model,
    *,
    seq_len=40,
    loss_fn=None,
    opt=None,
    clip=None,
    device=None,
    dtype="float32",
):
    loss_fn = nn.SoftmaxLoss() if loss_fn is None else loss_fn
    if opt is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    state = None
    for index in range(0, len(batches) - 1, seq_len):
        x, target = get_batch(batches, index, seq_len, device=device, dtype=dtype)
        logits, state = model(x, state)
        state = _detach_state(state)

        loss = loss_fn(logits, target)
        examples = target.shape[0]
        total_loss += float(loss.detach().numpy()) * examples
        total_correct += int(
            np.sum(
                np.argmax(logits.detach().numpy(), axis=1)
                == target.numpy().astype("int32")
            )
        )
        total_examples += examples

        if opt is not None:
            opt.zero_grad()
            loss.backward()
            if clip is not None:
                opt.clip_grad_norm(clip)
            opt.step()

    return total_correct / total_examples, total_loss / total_examples


__all__ = ["epoch"]
