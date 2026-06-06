import torch


def splice_cache(cache, new_values, start_idx, end_idx, grad_only_new_values=False):
    if end_idx - start_idx != new_values.shape[1]:
        raise ValueError("cache update range must match the new value length.")

    with torch.no_grad():
        cache[:, start_idx:end_idx].copy_(new_values)

    left = cache[:, :start_idx]
    right = cache[:, end_idx:]
    if grad_only_new_values:
        left = left.detach()
        right = right.detach()
    return torch.cat((left, new_values, right), dim=1)
