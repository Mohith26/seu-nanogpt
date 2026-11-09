"""Experiment 1: what does one bit flip do to a training run?

Train a clean model to step WARM, snapshot it, then for every (target, bit, rep)
restore the snapshot, flip one bit of one uniformly chosen float32 element in the
target (parameters, AdamW exp_avg, or AdamW exp_avg_sq), train HORIZON more steps
on the same data order, and compare validation loss with the clean continuation.

Resumable: results are appended to results/single_fault.jsonl, finished trials
are skipped. Usage: python experiments/single_fault.py [max_seconds]
"""
import json
import math
import os
import sys
import time
import zlib

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seu.core import CharData, TrainConfig, Trainer, bit_class, flip_bit_, pick_element, target_tensors  # noqa: E402

WARM, HORIZON, REPS, SEED = 1000, 200, 4, 0
TARGETS = ["param", "exp_avg", "exp_avg_sq"]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "single_fault.jsonl")


def main(max_seconds):
    torch.set_num_threads(4)
    t_start = time.time()
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["target"], r["bit"], r["rep"]))
    data = CharData()
    tr = Trainer(data, TrainConfig(), SEED)
    for _ in range(WARM):
        tr.train_step()
    snap = tr.snapshot()

    def run_horizon():
        worst = 0.0
        nonfinite = False
        for _ in range(HORIZON):
            loss, gnorm = tr.train_step()
            if not math.isfinite(loss):
                nonfinite = True
            else:
                worst = max(worst, loss)
        v = tr.val_loss()
        return v, nonfinite or not math.isfinite(v), worst

    clean_a, _, _ = run_horizon()
    tr.restore(snap)
    clean_b, _, _ = run_horizon()
    tr.restore(snap)
    clean_c, _, _ = run_horizon()
    assert clean_a == clean_b == clean_c, "snapshot/restore is not exact"
    print(f"warm={WARM} clean val after +{HORIZON}: {clean_a:.6f} / {clean_b:.6f} (determinism check)")

    with open(OUT, "a") as out:
        for target in TARGETS:
            for bit in range(32):
                for rep in range(REPS):
                    if (target, bit, rep) in done:
                        continue
                    if time.time() - t_start > max_seconds:
                        print("time budget reached; rerun to resume")
                        return
                    tr.restore(snap)
                    gen = torch.Generator().manual_seed(zlib.crc32(f"{target}/{bit}/{rep}".encode()))
                    t, k = pick_element(target_tensors(tr.model, tr.opt, target), gen)
                    old, new = flip_bit_(t, k, bit)
                    v, nonfinite, worst = run_horizon()
                    rec = {"target": target, "bit": bit, "bit_class": bit_class(bit), "rep": rep,
                           "old": old, "new": new, "val": v if math.isfinite(v) else None,
                           "clean_val": clean_a, "nonfinite": nonfinite,
                           "max_train_loss": worst}
                    out.write(json.dumps(rec) + "\n")
                    out.flush()


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1e9)
