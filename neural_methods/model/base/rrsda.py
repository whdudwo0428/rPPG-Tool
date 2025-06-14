"""
Adapted from here: https://github.com/rayleizhu/BiFormer
"""
import torch
from torch import Tensor, LongTensor , nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any


def _grid2seq(x:Tensor, region_size:Tuple[int], num_heads:int):
    """
    Args:
        x: BCTHW tensor
        region size: int
        num_heads: number of attention heads
    Return:
        out: rearranged x, has a shape of (bs, nhead, nregion, reg_size, head_dim)
        region_t, region_h, region_w: number of regions per t/col/row
    """
    B, C, T, H, W = x.size()
    region_t ,region_h, region_w = T//region_size[0],  H//region_size[1],  W//region_size[2]
    x = x.view(B, num_heads, C//num_heads, region_t, region_size[0],region_h, region_size[1], region_w, region_size[2])
    x = torch.einsum('bmdtohpwq->bmthwopqd', x).flatten(2, 4).flatten(-4, -2) # (bs, nhead, nregion, reg_size, head_dim)
    return x, region_t, region_h, region_w


def _seq2grid(x:Tensor, region_t:int, region_h:int, region_w:int, region_size:Tuple[int]):
    """
    Args: 
        x: (bs, nhead, nregion, reg_size^2, head_dim)
    Return:
        x: (bs, C, T, H, W)
    """
    bs, nhead, nregion, reg_size_square, head_dim = x.size()
    x = x.view(bs, nhead, region_t, region_h, region_w, region_size[0], region_size[1], region_size[2], head_dim)
    x = torch.einsum('bmthwopqd->bmdtohpwq', x).reshape(bs, nhead*head_dim,
        region_t*region_size[0],region_h*region_size[1], region_w*region_size[2])
    return x


def video_regional_routing_attention_torch(
    query:Tensor, key:Tensor, value:Tensor, scale:float,
    region_graph:LongTensor, region_size:Tuple[int],
    kv_region_size:Optional[Tuple[int]]=None,
    auto_pad=False)-> tuple[Any, Tensor]:
    """
    Args:
        query, key, value: (B, C, T, H, W) tensor
        scale: the scale/temperature for dot product attention
        region_graph: (B, nhead, t_q*h_q*w_q, topk) tensor, topk <= t_k*h_k*w_k
        region_size: region/window size for queries, (rt, rh, rw)
        key_region_size: optional, if None, key_region_size=region_size
    Return:
        output: (B, C, T, H, W) tensor
        attn: (bs, nhead, q_nregion, reg_size, topk*kv_region_size) attention matrix
        :param auto_pad:
        :param kv_region_size:
        :param region_size:
        :param region_graph:
        :param scale:
        :param value:
        :param key:
        :param query:
    """
    kv_region_size = kv_region_size or region_size
    bs, nhead, q_nregion, topk = region_graph.size()
    
    # # Auto pad to deal with any input size 
    # q_pad_b, q_pad_r, kv_pad_b, kv_pad_r = 0, 0, 0, 0
    # if auto_pad:
    #     _, _, Hq, Wq = query.size()
    #     q_pad_b = (region_size[0] - Hq % region_size[0]) % region_size[0]
    #     q_pad_r = (region_size[1] - Wq % region_size[1]) % region_size[1]
    #     if (q_pad_b > 0 or q_pad_r > 0):
    #         query = F.pad(query, (0, q_pad_r, 0, q_pad_b)) # zero padding

    #     _, _, Hk, Wk = key.size()
    #     kv_pad_b = (kv_region_size[0] - Hk % kv_region_size[0]) % kv_region_size[0]
    #     kv_pad_r = (kv_region_size[1] - Wk % kv_region_size[1]) % kv_region_size[1]
    #     if (kv_pad_r > 0 or kv_pad_b > 0):
    #         key = F.pad(key, (0, kv_pad_r, 0, kv_pad_b)) # zero padding
    #         value = F.pad(value, (0, kv_pad_r, 0, kv_pad_b)) # zero padding
    
    # to sequence format, i.e. (bs, nhead, nregion, reg_size, head_dim)
    query, q_region_t, q_region_h, q_region_w = _grid2seq(query, region_size=region_size, num_heads=nhead)
    key, _, _, _ = _grid2seq(key, region_size=kv_region_size, num_heads=nhead)
    value, _, _, _ = _grid2seq(value, region_size=kv_region_size, num_heads=nhead)

    # gather key and values.
    # torch.gather does not support broadcasting, hence we do it manually
    bs, nhead, kv_nregion, kv_region_size, head_dim = key.size()
    broadcasted_region_graph = region_graph.view(bs, nhead, q_nregion, topk, 1, 1).\
        expand(-1, -1, -1, -1, kv_region_size, head_dim)
    key_g = torch.gather(key.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim).\
        expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
        index=broadcasted_region_graph) # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)
    value_g = torch.gather(value.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim).\
        expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
        index=broadcasted_region_graph) # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)
    
    # token-to-token attention
    # (bs, nhead, q_nregion, reg_size, head_dim) @ (bs, nhead, q_nregion, head_dim, topk*kv_region_size)
    # -> (bs, nhead, q_nregion, reg_size, topk*kv_region_size)
    attn = (query * scale) @ key_g.flatten(-3, -2).transpose(-1, -2)
    attn = torch.softmax(attn, dim=-1)
    # (bs, nhead, q_nregion, reg_size, topk*kv_region_size) @ (bs, nhead, q_nregion, topk*kv_region_size, head_dim)
    # -> (bs, nhead, q_nregion, reg_size, head_dim)
    output = attn @ value_g.flatten(-3, -2)

    # to BCTHW format
    output = _seq2grid(output, region_t=q_region_t, region_h=q_region_h, region_w=q_region_w, region_size=region_size)

    # remove paddings if needed
    # if auto_pad and (q_pad_b > 0 or q_pad_r > 0):
    #     output = output[:, :, :Hq, :Wq]

    return output, attn
