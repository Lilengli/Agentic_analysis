"""CPU-only sanity check of the blockwise online-softmax recurrence."""
import math
import random

rng = random.Random(1234)
for length in (1, 8, 32, 33, 65, 128):
    for row in range(length):
        logits = [rng.uniform(-7, 7) for _ in range(row + 1)]
        values = [rng.uniform(-2, 2) for _ in logits]
        maximum = max(logits)
        denom = sum(math.exp(x - maximum) for x in logits)
        ref = sum(math.exp(x - maximum) * y for x, y in zip(logits, values)) / denom
        m, total, out = float('-inf'), 0.0, 0.0
        for start in range(0, row + 1, 32):
            x, y = logits[start:start + 32], values[start:start + 32]
            next_m = max(m, max(x))
            factor = math.exp(m - next_m)
            weights = [math.exp(z - next_m) for z in x]
            total = total * factor + sum(weights)
            out = out * factor + sum(w * z for w, z in zip(weights, y))
            m = next_m
        assert abs(out / total - ref) < 1e-12, (length, row)
print('Online-softmax math: PASS')
