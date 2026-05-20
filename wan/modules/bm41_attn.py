import os

import torch
import torch.nn.functional as F

__all__ = [
    "bm41_attention",
    "bm41_attn_matrix",
]


def _pad_tokens(x, block_size):
    pad = (-x.size(2)) % block_size
    if pad == 0:
        return x, pad
    return F.pad(x, (0, 0, 0, pad)), pad


def _key_masks(length, padded_length, block_size, device):
    mask = torch.arange(padded_length, device=device) < length
    block_mask = mask.view(padded_length // block_size, block_size)
    return block_mask, block_mask.any(dim=-1)


def _bm41_w_alpha(q, k, block_size, softmax_scale=None):
    B, H, Lq, D = q.shape
    _, _, Lk, _ = k.shape
    scale = D ** -0.5 if softmax_scale is None else softmax_scale

    q, q_pad = _pad_tokens(q, block_size)
    k, k_pad = _pad_tokens(k, block_size)
    q_len, k_len = Lq, Lk
    Lq_pad, Lk_pad = q.size(2), k.size(2)
    Rq, Rk = Lq_pad // block_size, Lk_pad // block_size

    qr = q.view(B, H, Rq, block_size, D)
    kr = k.view(B, H, Rk, block_size, D)

    key_token_mask, key_block_mask = _key_masks(
        k_len, Lk_pad, block_size, k.device)
    w_logits = torch.einsum(
        "bhrd,bhcsd->bhrcs", qr.mean(3), kr) * scale
    w_logits = w_logits.masked_fill(
        ~key_token_mask.view(1, 1, 1, Rk, block_size),
        torch.finfo(w_logits.dtype).min,
    )
    w = F.softmax(w_logits.float(), dim=-1).to(q.dtype)
    w = w.masked_fill(
        ~key_token_mask.view(1, 1, 1, Rk, block_size), 0)

    wf = w.float().clamp(min=1e-8)
    entropy = -(wf * wf.log()).sum(-1)
    entropy = entropy.masked_fill(
        ~key_block_mask.view(1, 1, 1, Rk), torch.finfo(entropy.dtype).min)

    weighted_k = torch.einsum("bhrcs,bhcsd->bhrcd", w, kr)
    alpha_logits = (
        torch.einsum("bhrtd,bhrcd->bhrtc", qr, weighted_k) * scale
        + entropy.unsqueeze(3)
    )
    alpha_logits = alpha_logits.masked_fill(
        ~key_block_mask.view(1, 1, 1, 1, Rk),
        torch.finfo(alpha_logits.dtype).min,
    )
    alpha = F.softmax(alpha_logits.float(), dim=-1).to(q.dtype)
    return qr, kr, w, alpha, q_pad, k_pad, q_len, k_len, scale


def bm41_attention(q, k, v, block_size, softmax_scale=None, query_block_chunk=None):
    """
    BM41 attention for tensors shaped [B, L, H, D].

    The implementation accepts different query/key lengths for KV-cache style
    calls. Padding is masked for keys and cropped from queries before returning.
    Query blocks are streamed in chunks so small block sizes do not materialize
    the full [query_blocks, key_blocks] BM41 state.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, and v must be rank-4 tensors [B, L, H, D]")
    if q.size(0) != k.size(0) or q.size(0) != v.size(0):
        raise ValueError("q, k, and v must have the same batch size")
    if q.size(2) != k.size(2) or q.size(2) != v.size(2):
        raise ValueError("q, k, and v must have the same number of heads")
    if q.size(3) != k.size(3):
        raise ValueError("q and k must have the same head dimension")
    if k.size(1) != v.size(1):
        raise ValueError("k and v must have the same sequence length")

    q_bhld = q.transpose(1, 2).contiguous()
    k_bhld = k.transpose(1, 2).contiguous()
    v_bhld = v.transpose(1, 2).contiguous()

    B, H, Lq, D = q_bhld.shape
    _, _, Lk, _ = k_bhld.shape
    value_dim = v.size(-1)
    scale = D ** -0.5 if softmax_scale is None else softmax_scale

    q_bhld, _ = _pad_tokens(q_bhld, block_size)
    k_bhld, k_pad = _pad_tokens(k_bhld, block_size)
    if k_pad:
        v_bhld = F.pad(v_bhld, (0, 0, 0, k_pad))
    Lq_pad, Lk_pad = q_bhld.size(2), k_bhld.size(2)
    Rq, Rk = Lq_pad // block_size, Lk_pad // block_size

    qr = q_bhld.view(B, H, Rq, block_size, D)
    kr = k_bhld.view(B, H, Rk, block_size, D)
    vr = v_bhld.view(B, H, Rk, block_size, value_dim)
    out_blocks = torch.empty(
        B, H, Rq, block_size, value_dim, device=q.device, dtype=v.dtype)

    if query_block_chunk is None:
        query_block_chunk = int(os.environ.get("BM41_QUERY_BLOCK_CHUNK", "64"))
    query_block_chunk = max(1, int(query_block_chunk))

    key_token_mask, key_block_mask = _key_masks(
        Lk, Lk_pad, block_size, k.device)
    key_token_mask = key_token_mask.view(1, 1, 1, Rk, block_size)
    key_block_mask = key_block_mask.view(1, 1, 1, Rk)

    for start in range(0, Rq, query_block_chunk):
        end = min(start + query_block_chunk, Rq)
        q_chunk = qr[:, :, start:end]
        q_mean = q_chunk.mean(3)

        w_logits = torch.einsum("bhqd,bhksd->bhqks", q_mean, kr) * scale
        w_logits = w_logits.masked_fill(
            ~key_token_mask, torch.finfo(w_logits.dtype).min)
        w = F.softmax(w_logits.float(), dim=-1).to(q.dtype)
        w = w.masked_fill(~key_token_mask, 0)

        wf = w.float().clamp(min=1e-8)
        entropy = -(wf * wf.log()).sum(-1)
        entropy = entropy.masked_fill(
            ~key_block_mask, torch.finfo(entropy.dtype).min)

        weighted_k = torch.einsum("bhqks,bhksd->bhqkd", w, kr)
        alpha_logits = (
            torch.einsum("bhqtd,bhqkd->bhqtk", q_chunk, weighted_k) * scale
            + entropy.unsqueeze(3)
        )
        alpha_logits = alpha_logits.masked_fill(
            ~key_block_mask.unsqueeze(3), torch.finfo(alpha_logits.dtype).min)
        alpha = F.softmax(alpha_logits.float(), dim=-1).to(q.dtype)

        weighted_v = torch.einsum("bhqks,bhksd->bhqkd", w, vr)
        out_blocks[:, :, start:end] = torch.einsum(
            "bhqtk,bhqkd->bhqtd", alpha, weighted_v)

    out = out_blocks.reshape(B, H, -1, value_dim)[:, :, :Lq]
    return out.transpose(1, 2).contiguous()


def bm41_attn_matrix(q, k, block_size, softmax_scale=None):
    """Return the amortized BM41 attention matrix for [B, H, L, D] inputs."""
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError("q and k must be rank-4 tensors [B, H, L, D]")
    _, _, w, alpha, _, _, q_len, k_len, _ = _bm41_w_alpha(
        q, k, block_size, softmax_scale=softmax_scale)
    attn = (alpha.unsqueeze(-1) * w.unsqueeze(3))
    B, H, _, block, _, _ = attn.shape
    return attn.reshape(B, H, -1, attn.size(4) * block)[:, :, :q_len, :k_len]
