import os

import torch
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    "bm41_attention",
    "bm41_attn_matrix",
]


def _rearrange_tokens(x, f_tied, h_reduce, w_reduce, h, w):
    """Arrange video tokens into exactly the block layout used by MonarchRT."""
    b, seq, num_heads, dim = x.shape
    block_tokens = f_tied * h * w
    if h % h_reduce or w % w_reduce:
        raise ValueError(
            f"h={h} and w={w} must be divisible by h_reduce={h_reduce} "
            f"and w_reduce={w_reduce}."
        )
    if seq % block_tokens:
        raise ValueError(
            f"sequence length {seq} must be divisible by "
            f"f_tied*h*w={block_tokens}."
        )

    x = x.view(
        b,
        -1,
        f_tied,
        h_reduce,
        h // h_reduce,
        w_reduce,
        w // w_reduce,
        num_heads,
        dim,
    )
    return rearrange(x, "b a f c i e j n d -> b (a c e) (f i) j n d")


def _restore_tokens(x, f_tied, h_reduce, w_reduce):
    return rearrange(
        x,
        "b (a c e) (f i) j n d -> b (a f c i e j) n d",
        c=h_reduce,
        e=w_reduce,
        f=f_tied,
    )


def _bm41_factors(q, k, scale):
    """
    Compute BM41 factors in the MonarchRT block family.

    q has shape [B, A, I, J, H, D] and k has shape [B, F, K, L, H, D].
    The Monarch rank-one block is indexed by (a, f, j, k), with row index i
    and column index l. BM41 differs from MRT-1 only by using mean_i(q) to
    initialize beta/R instead of the identity-selected q[..., k, j].
    """
    q_mean = q.mean(dim=2)
    beta_logits = torch.einsum("bajnd,bfklnd->bnafjkl", q_mean, k) * scale
    beta = F.softmax(beta_logits.float(), dim=-1).to(q.dtype)

    beta_float = beta.float().clamp_min(torch.finfo(torch.float32).tiny)
    entropy = -(beta_float * beta_float.log()).sum(dim=-1)
    compressed_k = torch.einsum("bnafjkl,bfklnd->bafjknd", beta, k)

    alpha_logits = (
        torch.einsum("baijnd,bafjknd->bnafjki", q, compressed_k) * scale
        + entropy.unsqueeze(-1)
    )
    alpha = rearrange(alpha_logits, "b n a f j k i -> b n a j i (f k)")
    alpha = F.softmax(alpha.float(), dim=-1).to(q.dtype)
    alpha = rearrange(
        alpha,
        "b n a j i (f k) -> b n a f j k i",
        f=k.size(1),
        k=k.size(2),
    )
    return beta, alpha


def _bm41_attention_blocks(q, k, v, scale, query_outer_chunk):
    b, outer_q, inner_i, inner_j, num_heads, value_dim = (
        q.size(0),
        q.size(1),
        q.size(2),
        q.size(3),
        v.size(-2),
        v.size(-1),
    )
    out = torch.empty(
        b,
        outer_q,
        inner_i,
        inner_j,
        num_heads,
        value_dim,
        device=v.device,
        dtype=v.dtype,
    )

    for start in range(0, outer_q, query_outer_chunk):
        end = min(start + query_outer_chunk, outer_q)
        beta, alpha = _bm41_factors(q[:, start:end], k, scale)
        compressed_v = torch.einsum("bnafjkl,bfklne->bafjkne", beta, v)
        out[:, start:end] = torch.einsum(
            "bnafjki,bafjkne->baijne", alpha, compressed_v
        )
    return out


def bm41_attention(
    q,
    k,
    v,
    f_tied,
    h_reduce,
    w_reduce,
    h,
    w,
    softmax_scale=None,
    query_outer_chunk=None,
):
    """
    BM41 attention for tensors shaped [B, L, H, D].

    The block layout is identical to ``monarch_attn``. Within each Monarch
    rank-one block, BM41 uses the mean query over the block row dimension to
    create one compressed key/value token, then attends to compressed tokens
    using the entropy correction from the variational objective.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, and v must be rank-4 tensors [B, L, H, D].")
    if q.size(0) != k.size(0) or q.size(0) != v.size(0):
        raise ValueError("q, k, and v must have the same batch size.")
    if q.size(2) != k.size(2) or q.size(2) != v.size(2):
        raise ValueError("q, k, and v must have the same number of heads.")
    if q.size(3) != k.size(3):
        raise ValueError("q and k must have the same head dimension.")
    if k.size(1) != v.size(1):
        raise ValueError("k and v must have the same sequence length.")
    if query_outer_chunk is None:
        query_outer_chunk = int(os.environ.get("BM41_QUERY_OUTER_CHUNK", "1"))
    query_outer_chunk = max(1, int(query_outer_chunk))

    scale = q.size(-1) ** -0.5 if softmax_scale is None else softmax_scale
    q_blocks = _rearrange_tokens(q, f_tied, h_reduce, w_reduce, h, w)
    k_blocks = _rearrange_tokens(k, f_tied, h_reduce, w_reduce, h, w)
    v_blocks = _rearrange_tokens(v, f_tied, h_reduce, w_reduce, h, w)
    out = _bm41_attention_blocks(
        q_blocks, k_blocks, v_blocks, scale, query_outer_chunk
    )
    return _restore_tokens(out, f_tied, h_reduce, w_reduce)


def _rearranged_token_order(length, f_tied, h_reduce, w_reduce, h, w, device):
    index = torch.arange(length, device=device).view(1, length, 1, 1)
    return _rearrange_tokens(
        index, f_tied, h_reduce, w_reduce, h, w
    ).reshape(-1).long()


def bm41_attn_matrix(
    q,
    k,
    f_tied,
    h_reduce,
    w_reduce,
    h,
    w,
    softmax_scale=None,
):
    """Return the BM41 attention matrix for [B, H, L, D] inputs."""
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError("q and k must be rank-4 tensors [B, H, L, D].")
    scale = q.size(-1) ** -0.5 if softmax_scale is None else softmax_scale
    q_blhd = q.transpose(1, 2).contiguous()
    k_blhd = k.transpose(1, 2).contiguous()
    q_blocks = _rearrange_tokens(q_blhd, f_tied, h_reduce, w_reduce, h, w)
    k_blocks = _rearrange_tokens(k_blhd, f_tied, h_reduce, w_reduce, h, w)
    beta, alpha = _bm41_factors(q_blocks, k_blocks, scale)

    ordered = rearrange(
        alpha.unsqueeze(-1) * beta.unsqueeze(-2),
        "b n a f j k i l -> b n (a i j) (f k l)",
    )
    q_order = _rearranged_token_order(
        q.size(2), f_tied, h_reduce, w_reduce, h, w, q.device
    )
    k_order = _rearranged_token_order(
        k.size(2), f_tied, h_reduce, w_reduce, h, w, k.device
    )
    q_inverse = torch.empty_like(q_order)
    k_inverse = torch.empty_like(k_order)
    q_inverse[q_order] = torch.arange(q_order.numel(), device=q.device)
    k_inverse[k_order] = torch.arange(k_order.numel(), device=k.device)
    return ordered[:, :, q_inverse][:, :, :, k_inverse]
