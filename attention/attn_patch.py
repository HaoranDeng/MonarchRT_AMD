import torch
import torch.nn.functional as F


def _splice_cache(cache, new_values, start_idx, end_idx, grad_only_new_values):
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


def full_attention_with_kv_cache(
    q,
    k_cache,
    v_cache,
    new_k,
    new_v,
    start_idx,
    end_idx,
    sm_scale=None,
    grad_only_new_kv=False,
):
    k = _splice_cache(k_cache, new_k, start_idx, end_idx, grad_only_new_kv)
    v = _splice_cache(v_cache, new_v, start_idx, end_idx, grad_only_new_kv)

    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
        scale=sm_scale,
    )
    return out.transpose(1, 2).contiguous()


__all__ = ["full_attention_with_kv_cache"]
