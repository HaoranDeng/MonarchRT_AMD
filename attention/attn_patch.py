import torch
import torch.nn.functional as F

from .cache import splice_cache


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
    k = splice_cache(k_cache, new_k, start_idx, end_idx, grad_only_new_kv)
    v = splice_cache(v_cache, new_v, start_idx, end_idx, grad_only_new_kv)

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
