"""CPU-only algebra/orchestration check, not a test of NPU compilation or execution."""
import numpy as np

rng = np.random.RandomState(41)
for seq in (128, 256, 384):
    for heads in (1, 2):
        q, k, v = [rng.randn(heads, seq, 128).astype(np.float16) for _ in range(3)]
        m = np.full((heads, seq), -np.finfo(np.float32).max, np.float32)
        l = np.zeros((heads, seq), np.float32)
        o = np.zeros((heads, seq, 128), np.float32)
        for start in range(0, seq, 128):
            scores = (q[:, start:, :].astype(np.float32) @ k[:, start:start+128, :].astype(np.float32).transpose(0, 2, 1)).astype(np.float16).astype(np.float32) / np.sqrt(128.)
            scores = np.where(np.arange(128)[None, None, :] <= np.arange(seq-start)[None, :, None], scores, -np.inf)
            m_next = np.maximum(m[:, start:], scores.max(axis=-1))
            p = np.exp(scores - m_next[:, :, None])
            alpha = np.exp(m[:, start:] - m_next)
            l[:, start:] = l[:, start:] * alpha + p.sum(axis=-1)
            o[:, start:, :] = o[:, start:, :] * alpha[:, :, None] + (p.astype(np.float16).astype(np.float32) @ v[:, start:start+128, :].astype(np.float32)).astype(np.float16).astype(np.float32)
            m[:, start:] = m_next
        scores = (q.astype(np.float32) @ k.astype(np.float32).transpose(0, 2, 1)) / np.sqrt(128.)
        scores = np.where(np.arange(seq)[None, None, :] <= np.arange(seq)[None, :, None], scores, -np.inf)
        p = np.exp(scores - scores.max(axis=-1, keepdims=True))
        ref = p @ v.astype(np.float32) / p.sum(axis=-1, keepdims=True)
        actual = o / l[:, :, None]
        error = np.max(np.abs(actual - ref))
        assert np.allclose(actual, ref, atol=.002, rtol=.01), error
        print(f'seq={seq} heads={heads} max_abs={error:.8g}')
