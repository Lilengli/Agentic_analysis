"""CPU tests of the actual Python tile orchestration (not the Ascend C kernel)."""
import math
import unittest
from unittest.mock import patch
import torch
from tiled_attention import causal_attention


class TiledAttentionTests(unittest.TestCase):
    def test_state_update_and_causal_reference(self):
        torch.set_num_threads(1)
        for seq in (128, 256, 384):
            for seed in (17, 1234):
                with self.subTest(seq=seq, seed=seed), torch.inference_mode():
                    torch.manual_seed(seed)
                    q, k, v = [torch.randn((2, 2, seq, 128), dtype=torch.float16) for _ in range(3)]
                    concat = causal_attention(q, k, v, cpu_reference_stage=True, state_update='concat')
                    with patch('torch.cat', side_effect=AssertionError('inplace path must not concatenate state')):
                        inplace = causal_attention(q, k, v, cpu_reference_stage=True)
                    torch.testing.assert_close(inplace, concat, atol=0, rtol=0)
                    scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(128)
                    mask = torch.ones(seq, seq, dtype=torch.bool).triu(1)
                    expected = torch.softmax(scores.masked_fill(mask, -float('inf')), -1) @ v.float()
                    torch.testing.assert_close(inplace.float(), expected, atol=.002, rtol=.01)
                    changed = v.clone()
                    changed[:, :, seq // 2:, :] += 1
                    perturbed = causal_attention(q, k, changed, cpu_reference_stage=True)
                    torch.testing.assert_close(perturbed[:, :, :seq // 2], inplace[:, :, :seq // 2], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
