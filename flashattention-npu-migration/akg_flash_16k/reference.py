"""AKG task: contiguous BNSD FP16 causal prefill attention."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)


def get_inputs():
    torch.manual_seed(1234)
    return [torch.randn((1, 2, 16384, 128), dtype=torch.float16, device='npu')
            for _ in range(3)]


def get_init_inputs():
    return []
