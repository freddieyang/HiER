"""Data-independent whole-view masking schedules for controlled training."""
import math


def morgan_probability_for_epoch(policy: str, initial: float, epoch: int, epochs: int) -> float:
    """Return an explicit per-epoch probability without reading metrics or labels."""
    if not math.isfinite(initial) or not 0 <= initial <= 1:
        raise ValueError('Initial probability must be finite and in [0, 1]')
    if epochs < 1 or not 0 <= epoch < epochs:
        raise ValueError('Epoch must be within the declared training budget')
    if policy == 'constant':
        return float(initial)
    if policy == 'linear_decay':
        if epochs < 2:
            raise ValueError('Linear decay needs at least two epochs')
        return float(initial * (1.0 - epoch / (epochs - 1)))
    raise ValueError(f'Unknown masking schedule: {policy}')
