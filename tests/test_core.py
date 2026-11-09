import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seu.core import CharData, Guard, TrainConfig, Trainer, bit_class, flip_bit_  # noqa: E402

SMALL = TrainConfig(n_layer=1, n_head=2, n_embd=32, batch_size=4, block_size=16, eval_batches=2)


class FlipTest(unittest.TestCase):
    def test_flip_known_bits(self):
        t = torch.tensor([1.0, 1.0, 1.0, 0.5])
        self.assertEqual(flip_bit_(t, 0, 31), (1.0, -1.0))           # sign
        self.assertEqual(flip_bit_(t, 1, 23)[1], 0.5)                # exponent LSB: 1.0 -> 0.5
        self.assertTrue(math.isinf(flip_bit_(t, 2, 30)[1]))          # 1.0 has exponent 127; bit 30 -> 255 = inf
        self.assertEqual(flip_bit_(t, 3, 22)[1], 0.75)               # mantissa MSB
        self.assertEqual(t[0].item(), -1.0)

    def test_flip_twice_restores(self):
        t = torch.randn(10)
        before = t.clone()
        flip_bit_(t, 7, 13)
        flip_bit_(t, 7, 13)
        self.assertTrue(torch.equal(t, before))

    def test_bit_class(self):
        self.assertEqual([bit_class(b) for b in (0, 22, 23, 30, 31)],
                         ["mantissa", "mantissa", "exponent", "exponent", "sign"])


class TrainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = CharData()

    def test_restore_is_exact_and_does_not_alias(self):
        tr = Trainer(self.data, SMALL, seed=0)
        for _ in range(5):
            tr.train_step()
        snap = tr.snapshot()
        losses_a = [tr.train_step()[0] for _ in range(5)]
        tr.restore(snap)
        losses_b = [tr.train_step()[0] for _ in range(5)]
        tr.restore(snap)
        losses_c = [tr.train_step()[0] for _ in range(5)]
        self.assertEqual(losses_a, losses_b)
        self.assertEqual(losses_b, losses_c)

    def test_guard_rolls_back_after_poisoned_moment(self):
        tr = Trainer(self.data, SMALL, seed=0)
        g = Guard(snapshot_every=5, check_every=1)
        g.snaps.append((tr.snapshot(), None))
        for _ in range(10):
            loss, gn = tr.train_step()
            self.assertIsNone(g.after_step(tr, loss, gn))
        p = next(tr.model.parameters())
        v = tr.opt.state[p]["exp_avg_sq"]
        flip_bit_(v, 0, 31)                      # second moment becomes negative
        loss, gn = tr.train_step()
        reason = g.after_step(tr, loss, gn)
        self.assertIsNotNone(reason)
        restored = g.roll_back(tr, reason, None)
        self.assertEqual(restored, 10)
        self.assertTrue(all(torch.isfinite(q).all() for q in tr.model.parameters()))
        self.assertTrue(bool((tr.opt.state[p]["exp_avg_sq"] >= 0).all()))


if __name__ == "__main__":
    unittest.main()
