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


def _rearrange_query_tokens_contiguous(x, q_block_size):
    b, seq, num_heads, dim = x.shape
    if seq % q_block_size:
        raise ValueError(
            f"sequence length {seq} must be divisible by q_block_size={q_block_size}."
        )
    return x.view(b, seq // q_block_size, q_block_size, 1, num_heads, dim)


def _restore_query_tokens_contiguous(x):
    b, outer, inner, one, num_heads, dim = x.shape
    if one != 1:
        raise ValueError("contiguous query layout expects a singleton j dimension.")
    return x.view(b, outer * inner, num_heads, dim)


def _random_q_indices(num_choices, num_positions, device, random_seed):
    if random_seed is None:
        return torch.randint(num_choices, (num_positions,), device=device)

    try:
        gen = torch.Generator(device=device)
    except (TypeError, RuntimeError):
        gen = torch.Generator()
    gen.manual_seed(int(random_seed))
    try:
        return torch.randint(
            num_choices, (num_positions,), device=device, generator=gen
        )
    except RuntimeError:
        return torch.randint(num_choices, (num_positions,), generator=gen).to(device)


def _initial_r_query(q, k_size, q_init, random_seed):
    q_init = q_init.lower()
    if q_init in {"identity", "ith", "i"}:
        return q
    if q_init in {"uniform", "mean", "avg", "average"}:
        return q.mean(dim=2, keepdim=True).expand(-1, -1, k_size, -1, -1, -1)
    if q_init in {"first", "1st"}:
        return q[:, :, :1].expand(-1, -1, k_size, -1, -1, -1)
    if q_init == "random":
        idx = _random_q_indices(q.size(2), k_size, q.device, random_seed)
        return q.index_select(2, idx)
    raise ValueError(
        f"unsupported q_init={q_init!r}; expected mean, random, 1st, or ith."
    )


def _monarch_one_step_factors(q, k, scale, q_init, random_seed=None):
    """
    Compute one MonarchAttention R/L update with a chosen initial R query.

    q has shape [B, A, I, J, H, D] and k has shape [B, F, K, L, H, D].
    Standard MRT-1 uses identity L^(0), so a_R[..., k, j] is q[..., k, j].
    BM41 changes only L^(0) to uniform along I, so each k uses mean_I(q).
    """
    if q.size(2) != k.size(2):
        raise ValueError(
            "one-step Monarch initialization requires matching I/K block sizes."
        )
    a_r = _initial_r_query(q, k.size(2), q_init, random_seed)

    r_logits = torch.einsum("bakjnd,bfklnd->bnafjkl", a_r, k) * scale
    r = F.softmax(r_logits.float(), dim=-1).to(q.dtype)

    r_float = r.float().clamp_min(torch.finfo(torch.float32).tiny)
    c_l = (r_float * r_float.log()).sum(dim=-1)
    a_l = torch.einsum("bnafjkl,bfklnd->bafjknd", r, k)

    l_logits = (
        torch.einsum("baijnd,bafjknd->bnafjki", q, a_l) * scale
        - c_l.unsqueeze(-1)
    )
    l = rearrange(l_logits, "b n a f j k i -> b n a j i (f k)")
    l = F.softmax(l.float(), dim=-1).to(q.dtype)
    l = rearrange(
        l,
        "b n a j i (f k) -> b n a f j k i",
        f=k.size(1),
        k=k.size(2),
    )
    return r, l


def _bm41_factors(q, k, scale, q_init, random_seed=None):
    return _monarch_one_step_factors(q, k, scale, q_init, random_seed=random_seed)


def _bm41_attention_blocks(q, k, v, scale, query_outer_chunk, q_init, random_seed):
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
        r, l = _bm41_factors(
            q[:, start:end], k, scale, q_init, random_seed=random_seed
        )
        compressed_v = torch.einsum("bnafjkl,bfklne->bafjkne", r, v)
        out[:, start:end] = torch.einsum(
            "bnafjki,bafjkne->baijne", l, compressed_v
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
    q_init=None,
    random_seed=None,
    layout=None,
):
    """
    BM41 attention for tensors shaped [B, L, H, D].

    The block layout and one-step R/L updates are identical to ``monarch_attn``.
    ``q_init`` controls the initial R query: mean, random, 1st, or ith.
    The ``ith`` mode matches MRT-1's identity L^(0) choice.
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
    q_init = os.environ.get("BM41_Q_INIT", q_init or "mean")
    layout = os.environ.get("BM41_LAYOUT", layout or "monarch")
    if "BM41_RANDOM_SEED" in os.environ:
        random_seed = int(os.environ["BM41_RANDOM_SEED"])

    scale = q.size(-1) ** -0.5 if softmax_scale is None else softmax_scale
    if layout == "monarch":
        q_blocks = _rearrange_tokens(q, f_tied, h_reduce, w_reduce, h, w)
        restore = _restore_tokens
    elif layout in {"contiguous_q", "contiguous"}:
        q_block_size = f_tied * (h // h_reduce)
        q_blocks = _rearrange_query_tokens_contiguous(q, q_block_size)
        restore = None
    else:
        raise ValueError(
            f"unsupported BM41 layout={layout!r}; expected monarch or contiguous_q."
        )
    k_blocks = _rearrange_tokens(k, f_tied, h_reduce, w_reduce, h, w)
    v_blocks = _rearrange_tokens(v, f_tied, h_reduce, w_reduce, h, w)
    out = _bm41_attention_blocks(
        q_blocks, k_blocks, v_blocks, scale, query_outer_chunk, q_init, random_seed
    )
    if restore is None:
        return _restore_query_tokens_contiguous(out)
    return restore(out, f_tied, h_reduce, w_reduce)


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
    q_init=None,
    random_seed=None,
):
    """Return the BM41 attention matrix for [B, H, L, D] inputs."""
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError("q and k must be rank-4 tensors [B, H, L, D].")
    q_init = os.environ.get("BM41_Q_INIT", q_init or "mean")
    if "BM41_RANDOM_SEED" in os.environ:
        random_seed = int(os.environ["BM41_RANDOM_SEED"])
    scale = q.size(-1) ** -0.5 if softmax_scale is None else softmax_scale
    q_blhd = q.transpose(1, 2).contiguous()
    k_blhd = k.transpose(1, 2).contiguous()
    q_blocks = _rearrange_tokens(q_blhd, f_tied, h_reduce, w_reduce, h, w)
    k_blocks = _rearrange_tokens(k_blhd, f_tied, h_reduce, w_reduce, h, w)
    r, l = _bm41_factors(q_blocks, k_blocks, scale, q_init, random_seed=random_seed)

    ordered = rearrange(
        l.unsqueeze(-1) * r.unsqueeze(-2),
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
