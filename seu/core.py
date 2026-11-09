"""Training loop, single-event-upset (bit flip) injection, and a rollback guard."""
import copy
import math
import os
import struct
import time
from dataclasses import dataclass, field

import torch

from .model import GPT, GPTConfig

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "input.txt")


# ---------------------------------------------------------------- data

class CharData:
    """tiny-shakespeare, character level, 90/10 split (as in nanoGPT's prepare.py)."""

    def __init__(self, path=DATA):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        stoi = {c: i for i, c in enumerate(chars)}
        ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(ids))
        self.train, self.val = ids[:n], ids[n:]

    def batch(self, split, batch_size, block_size, gen):
        d = self.train if split == "train" else self.val
        ix = torch.randint(len(d) - block_size - 1, (batch_size,), generator=gen)
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + block_size] for i in ix])
        return x, y


# ---------------------------------------------------------------- bit flips

def flip_bit_(t: torch.Tensor, flat_index: int, bit: int):
    """Flip one bit (0 = mantissa LSB, 31 = sign) of one float32 element in place.
    Returns (old_value, new_value)."""
    assert t.dtype == torch.float32
    flat = t.view(-1)
    old = float(flat[flat_index])
    as_int = struct.unpack("<I", struct.pack("<f", old))[0] ^ (1 << bit)
    new = struct.unpack("<f", struct.pack("<I", as_int))[0]
    with torch.no_grad():
        flat[flat_index] = new
    return old, new


def bit_class(bit: int) -> str:
    if bit == 31:
        return "sign"
    if bit >= 23:
        return "exponent"
    return "mantissa"


def target_tensors(model, opt, target):
    """The memory a fault can land in: model parameters or AdamW's two moment buffers."""
    if target == "param":
        return [p.data for p in model.parameters()]
    key = {"exp_avg": "exp_avg", "exp_avg_sq": "exp_avg_sq"}[target]
    return [opt.state[p][key] for p in model.parameters() if p in opt.state]


def pick_element(tensors, gen):
    """Pick one element uniformly over all elements of the given tensors."""
    sizes = [t.numel() for t in tensors]
    total = sum(sizes)
    k = int(torch.randint(total, (1,), generator=gen))
    for t, s in zip(tensors, sizes):
        if k < s:
            return t, k
        k -= s
    raise AssertionError


# ---------------------------------------------------------------- training state

@dataclass
class TrainConfig:
    batch_size: int = 12
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    lr: float = 1e-3
    warmup: int = 50
    beta2: float = 0.99
    weight_decay: float = 0.1
    eval_batches: int = 16


class Trainer:
    def __init__(self, data: CharData, cfg: TrainConfig, seed: int):
        torch.manual_seed(seed)
        self.data, self.cfg = data, cfg
        mcfg = GPTConfig(block_size=cfg.block_size, vocab_size=data.vocab_size,
                         n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd)
        self.model = GPT(mcfg)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                     betas=(0.9, cfg.beta2), weight_decay=cfg.weight_decay,
                                     foreach=False)
        self.gen = torch.Generator().manual_seed(seed + 1)
        self.step = 0

    # -- snapshots (used for experiment branching and by the guard)
    def snapshot(self):
        return {
            "model": copy.deepcopy(self.model.state_dict()),
            "opt": copy.deepcopy(self.opt.state_dict()),
            "gen": self.gen.get_state().clone(),
            "step": self.step,
        }

    def restore(self, snap):
        # Optimizer.load_state_dict keeps the given tensors when dtype/device
        # already match, so without a copy later in-place AdamW updates would
        # silently modify the snapshot itself.
        self.model.load_state_dict(copy.deepcopy(snap["model"]))
        self.opt.load_state_dict(copy.deepcopy(snap["opt"]))
        self.gen.set_state(snap["gen"].clone())
        self.step = snap["step"]

    def lr_at(self, step):
        return self.cfg.lr * min(1.0, (step + 1) / self.cfg.warmup)

    def train_step(self):
        """One optimizer step. Returns (loss, grad_norm) as floats."""
        self.model.train()
        for g in self.opt.param_groups:
            g["lr"] = self.lr_at(self.step)
        x, y = self.data.batch("train", self.cfg.batch_size, self.cfg.block_size, self.gen)
        _, loss = self.model(x, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.step += 1
        return float(loss.detach()), float(gnorm)

    @torch.no_grad()
    def val_loss(self):
        """Validation loss on a fixed set of batches (same batches every call)."""
        self.model.eval()
        gen = torch.Generator().manual_seed(12345)
        losses = []
        for _ in range(self.cfg.eval_batches):
            x, y = self.data.batch("val", self.cfg.batch_size, self.cfg.block_size, gen)
            _, loss = self.model(x, y)
            losses.append(float(loss))
        return sum(losses) / len(losses)


# ---------------------------------------------------------------- guard

@dataclass
class Guard:
    """Cheap detect-and-roll-back protection for a training loop.

    Every step it checks that the loss and gradient norm are finite and that the
    loss has not jumped far above its running average; every `check_every` steps it
    also checks that all parameters and AdamW moments are finite and that the
    second moment is non-negative (it must be, by construction). On a trip it
    restores the most recent snapshot (taken every `snapshot_every` steps, kept in
    RAM) and replays from there; if the same snapshot trips again it falls back to
    the older one.
    """
    snapshot_every: int = 50
    check_every: int = 10
    spike_factor: float = 1.5
    ema_beta: float = 0.9
    ema: float = None
    snaps: list = field(default_factory=list)
    rollbacks: int = 0
    trips: dict = field(default_factory=dict)

    def state_ok(self, tr: Trainer):
        for p in tr.model.parameters():
            if not torch.isfinite(p).all():
                return "param_nonfinite"
            st = tr.opt.state.get(p)
            if st:
                if not torch.isfinite(st["exp_avg"]).all():
                    return "exp_avg_nonfinite"
                v = st["exp_avg_sq"]
                if not torch.isfinite(v).all() or bool((v < 0).any()):
                    return "exp_avg_sq_invalid"
        return None

    def after_step(self, tr: Trainer, loss: float, gnorm: float):
        """Returns a trip reason, or None if the step looks healthy."""
        reason = None
        if not math.isfinite(loss) or not math.isfinite(gnorm):
            reason = "nonfinite_loss_or_grad"
        elif self.ema is not None and tr.step > 100 and loss > self.spike_factor * self.ema:
            reason = "loss_spike"
        elif tr.step % self.check_every == 0:
            reason = self.state_ok(tr)
        if reason is None:
            self.ema = loss if self.ema is None else self.ema_beta * self.ema + (1 - self.ema_beta) * loss
            if tr.step % self.snapshot_every == 0:
                self.snaps.append((tr.snapshot(), self.ema))
                self.snaps = self.snaps[-2:]
        return reason

    def roll_back(self, tr: Trainer, reason: str, last_restored_step):
        self.rollbacks += 1
        self.trips[reason] = self.trips.get(reason, 0) + 1
        snap, ema = self.snaps[-1]
        if last_restored_step == snap["step"] and len(self.snaps) > 1:
            self.snaps.pop()  # the newest snapshot itself looks bad: use the older one
            snap, ema = self.snaps[-1]
        tr.restore(snap)
        self.ema = ema
        return snap["step"]


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0
