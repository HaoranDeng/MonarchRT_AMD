import os

import torch
from einops import rearrange


def _is_identity_q_init(q_init):
    return q_init.lower() in {"identity", "ith", "i"}


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
    if _is_identity_q_init(q_init):
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


def _get_rearrange_fns(x, f_tied, h_reduce, w_reduce, h, w):
    b, _, nh, d = x.shape

    def rearrange_fn(x):
        x = x.view(
            b,
            -1,
            f_tied,
            h_reduce,
            h // h_reduce,
            w_reduce,
            w // w_reduce,
            nh,
            d,
        )
        return rearrange(x, "b a f c i e j h d -> b (a c e) (f i) j h d")

    def return_fn(x):
        return rearrange(
            x,
            "b (a c e) (f i) j h d -> b (a f c i e j) h d",
            c=h_reduce,
            e=w_reduce,
            f=f_tied,
        )

    return rearrange_fn, return_fn


def _monarch_one_step_factors(q, k, scale, q_init, random_seed=None):
    """
    Compute one MonarchAttention R/L update with a configurable L^(0).

    q has shape [B, A, I, J, H, D] and k has shape [B, F, K, L, H, D].
    Standard MRT-1 is q_init="ith": the initial L is identity, so
    a_R[..., k, j] is Q[..., k, j].
    """
    if q.size(2) != k.size(2):
        raise ValueError(
            "one-step Monarch initialization requires matching I/K block sizes."
        )

    sm_scale_sqrt = scale**0.5
    q_scaled = q * sm_scale_sqrt
    k_scaled = k * sm_scale_sqrt
    a_r = _initial_r_query(q_scaled, k_scaled.size(2), q_init, random_seed)

    r_logits = torch.einsum("bakjhd,bfklhd->bhafkjl", a_r, k_scaled)
    r_logits = r_logits - r_logits.amax(dim=-1, keepdim=True)
    r = torch.softmax(r_logits.float(), dim=-1).to(q.dtype)

    r_float = r.float().clamp_min(torch.finfo(torch.float32).tiny)
    c_l = (r_float * r_float.log()).sum(dim=-1, keepdim=True).transpose(-2, -3)
    a_l = torch.einsum("bhafkjl,bfklhd->bafjkhd", r, k_scaled)

    l_logits = torch.einsum("bafjkhd,baijhd->bhafjki", a_l, q_scaled) - c_l
    l = rearrange(l_logits, "b h a f j k i -> b h a j i (f k)")
    l = torch.softmax(l.float(), dim=-1).to(q.dtype)
    l = rearrange(l, "b h a j i (f k) -> b h a f j k i", f=k.size(1), k=k.size(2))
    return r, l


def _monarch_one_step_attention_blocks(
    q, k, v, scale, query_outer_chunk, q_init, random_seed
):
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
        r, l = _monarch_one_step_factors(
            q[:, start:end], k, scale, q_init, random_seed=random_seed
        )
        y = torch.einsum("bhafkjl,bfklhe->bafjkhe", r, v)
        out[:, start:end] = torch.einsum("bhafjki,bafjkhe->baijhe", l, y)
    return out


def monarch_attn_slow(q, k, v, sm_scale, num_iters=1):
    b, a, i, j, h, _ = q.shape
    block_b1, block_b2 = i, j
    num_k_blocks = k.shape[1]

    sm_scale_sqrt = sm_scale**0.5
    q = q * sm_scale_sqrt
    k = k * sm_scale_sqrt

    a_r = q.clone().unsqueeze(-5).expand(
        -1, -1, num_k_blocks, -1, -1, -1, -1
    )
    c_r = torch.ones(
        (b, h, a, num_k_blocks, block_b1, block_b2, 1),
        device=q.device,
        dtype=q.dtype,
    )

    for _ in range(num_iters - 1):
        b_r = torch.einsum("bafkjhd,bfklhd->bhafkjl", a_r, k)
        z = b_r.float() * (1.0 / (c_r + 1e-6)).clamp_max(1e4)
        z = z - z.amax(dim=-1, keepdim=True)
        r = torch.softmax(z, dim=-1).to(q.dtype)
        a_l = torch.einsum("bhafkjl,bfklhd->bafjkhd", r, k)
        logz = torch.logsumexp(z, dim=-1, keepdim=True)
        c_l = (r * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3)

        b_l = torch.einsum("bafjkhd,baijhd->bhafjki", a_l, q)
        l = rearrange(b_l - c_l, "b h a f j k i -> b h a j i (f k)")
        l = torch.softmax(l, dim=-1).to(q.dtype)
        l = rearrange(
            l,
            "b h a j i (f k) -> b h a f j k i",
            f=num_k_blocks,
            k=block_b1,
        )

        a_r = torch.einsum("bhafjki,baijhd->bafkjhd", l, q)
        c_r = l.sum(dim=-1, dtype=torch.float32).unsqueeze(-1).transpose(-2, -3)

    b_r = torch.einsum("bafkjhd,bfklhd->bhafkjl", a_r, k)
    z = b_r.float() * (1.0 / (c_r + 1e-6)).clamp_max(1e4)
    z = z - z.amax(dim=-1, keepdim=True)
    r = torch.softmax(z, dim=-1).to(q.dtype)
    a_l = torch.einsum("bhafkjl,bfklhd->bafjkhd", r, k)
    logz = torch.logsumexp(z, dim=-1, keepdim=True)
    c_l = (r * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3)
    y = torch.einsum("bhafkjl,bfklhd->bafjkhd", r, v)

    b_l = torch.einsum("bafjkhd,baijhd->bhafjki", a_l, q)
    l = rearrange(b_l - c_l, "b h a f j k i -> b h a j i (f k)")
    l = torch.softmax(l, dim=-1).to(q.dtype)
    l = rearrange(
        l,
        "b h a j i (f k) -> b h a f j k i",
        f=num_k_blocks,
        k=block_b1,
    )
    return torch.einsum("bhafjki,bafjkhd->baijhd", l, y)


def _resolve_q_init(q_init):
    return os.environ.get("MONARCH_Q_INIT", q_init or "ith")


def _resolve_random_seed(random_seed):
    if "MONARCH_RANDOM_SEED" in os.environ:
        return int(os.environ["MONARCH_RANDOM_SEED"])
    return random_seed


def _resolve_query_outer_chunk(query_outer_chunk):
    if query_outer_chunk is None:
        query_outer_chunk = int(os.environ.get("MONARCH_QUERY_OUTER_CHUNK", "1"))
    return max(1, int(query_outer_chunk))


def monarch_attn(
    q,
    k,
    v,
    f_tied,
    h_reduce,
    w_reduce,
    h,
    w,
    sm_scale=None,
    block_causal_size=None,
    num_iters=1,
    q_init=None,
    random_seed=None,
    query_outer_chunk=None,
):
    b, qs, nh, d = q.shape
    ks = k.shape[1]
    if sm_scale is None:
        sm_scale = d**-0.5

    q_init = _resolve_q_init(q_init)
    random_seed = _resolve_random_seed(random_seed)
    query_outer_chunk = _resolve_query_outer_chunk(query_outer_chunk)

    if block_causal_size is not None:
        block_tokens = f_tied * h * w
        if qs != ks:
            raise ValueError("causal Monarch attention requires equal q/k lengths.")
        if qs % block_causal_size or block_causal_size % block_tokens:
            raise ValueError("block_causal_size must align with Monarch block sizes.")
        chunks = []
        for end in range(block_causal_size, qs + 1, block_causal_size):
            start = end - block_causal_size
            chunks.append(
                monarch_attn(
                    q[:, start:end],
                    k[:, :end],
                    v[:, :end],
                    f_tied,
                    h_reduce,
                    w_reduce,
                    h,
                    w,
                    sm_scale=sm_scale,
                    num_iters=num_iters,
                    q_init=q_init,
                    random_seed=random_seed,
                    query_outer_chunk=query_outer_chunk,
                )
            )
        return torch.cat(chunks, dim=1)

    rearrange_fn, return_fn = _get_rearrange_fns(q, f_tied, h_reduce, w_reduce, h, w)
    q_blocks = rearrange_fn(q).contiguous()
    k_blocks = rearrange_fn(k).contiguous()
    v_blocks = rearrange_fn(v).contiguous()

    if num_iters == 1:
        out = _monarch_one_step_attention_blocks(
            q_blocks,
            k_blocks,
            v_blocks,
            sm_scale,
            query_outer_chunk,
            q_init,
            random_seed,
        )
    else:
        if not _is_identity_q_init(q_init):
            raise ValueError("non-identity Monarch q_init is only supported for num_iters=1.")
        out = monarch_attn_slow(q_blocks, k_blocks, v_blocks, sm_scale, num_iters)
    return return_fn(out)


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


def monarch_attn_with_kv_cache(
    q,
    k_cache,
    v_cache,
    new_k,
    new_v,
    start_idx,
    end_idx,
    f_tied,
    h_reduce,
    w_reduce,
    h,
    w,
    sm_scale=None,
    num_iters=1,
    q_init=None,
    random_seed=None,
    query_outer_chunk=None,
):
    grad_only_new_kv = torch.is_grad_enabled() and not (
        k_cache.requires_grad or v_cache.requires_grad
    )
    k = _splice_cache(k_cache, new_k, start_idx, end_idx, grad_only_new_kv)
    v = _splice_cache(v_cache, new_v, start_idx, end_idx, grad_only_new_kv)
    return monarch_attn(
        q,
        k,
        v,
        f_tied,
        h_reduce,
        w_reduce,
        h,
        w,
        sm_scale=sm_scale,
        num_iters=num_iters,
        q_init=q_init,
        random_seed=random_seed,
        query_outer_chunk=query_outer_chunk,
    )


__all__ = ["monarch_attn", "monarch_attn_with_kv_cache"]
