"""Experiment 2: full training runs under an accelerated upset rate, with and
without the rollback guard.

Each run trains STEPS steps from scratch. Upsets arrive as a Poisson process with
mean one per MEAN_GAP steps; each one flips a uniformly random bit of a uniformly
random float32 element of the training state (parameters + both AdamW moments,
weighted by size, i.e. by bytes of memory). The same seed gives the same model
init, data order and upset schedule with or without the guard. A fault-free run
per seed is the reference.

The upset rate is deliberately far above any real orbit rate so that each run sees
dozens of upsets; see README.

Resumable: results appended to results/guard.jsonl. Usage:
    python experiments/guard.py [max_seconds]
"""
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seu.core import CharData, Guard, TrainConfig, Trainer, bit_class, flip_bit_, pick_element, target_tensors  # noqa: E402

STEPS, MEAN_GAP, SEEDS = 2000, 25, [1, 2, 3, 4, 5, 6]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "guard.jsonl")


def upset_schedule(seed):
    g = torch.Generator().manual_seed(10_000 + seed)
    steps, s = [], 0.0
    while True:
        s += float(torch.empty(1).exponential_(1.0 / MEAN_GAP, generator=g))
        if s >= STEPS:
            return steps, g
        steps.append(int(s))


def run(data, seed, mode):
    """mode: clean | clean+guard | faults | faults+guard"""
    tr = Trainer(data, TrainConfig(), seed)
    sched, fault_gen = upset_schedule(seed)
    sched = sorted(sched) if mode.startswith("faults") else []
    fired = set()  # an upset is a physical event: it happens once even if we replay
    guard = Guard() if mode.endswith("guard") else None
    if guard:
        guard.snaps.append((tr.snapshot(), None))
    classes = {}
    restored_at = None
    replayed = 0
    diverged_at = None
    t0 = time.perf_counter()
    executed = 0
    while tr.step < STEPS:
        for i, s in enumerate(sched):
            if s == tr.step and i not in fired:
                fired.add(i)
                pool = ["param", "exp_avg", "exp_avg_sq"]
                tensors = [t for name in pool for t in target_tensors(tr.model, tr.opt, name)]
                t, k = pick_element(tensors, fault_gen)
                bit = int(torch.randint(32, (1,), generator=fault_gen))
                flip_bit_(t, k, bit)
                classes[bit_class(bit)] = classes.get(bit_class(bit), 0) + 1
        loss, gnorm = tr.train_step()
        executed += 1
        if guard:
            reason = guard.after_step(tr, loss, gnorm)
            if reason:
                before = tr.step
                restored_at = guard.roll_back(tr, reason, restored_at)
                replayed += before - tr.step
                if guard.rollbacks > 200:
                    break
        elif not math.isfinite(loss) and diverged_at is None:
            diverged_at = tr.step
    wall = time.perf_counter() - t0
    v = tr.val_loss()
    return {"seed": seed, "mode": mode, "val": v if math.isfinite(v) else None,
            "upsets": len(fired), "upset_bit_classes": classes,
            "diverged_at": diverged_at, "rollbacks": guard.rollbacks if guard else 0,
            "trips": guard.trips if guard else {}, "replayed_steps": replayed,
            "executed_steps": executed, "wall_s": wall, "ms_per_executed_step": 1000 * wall / executed}


def main(max_seconds):
    torch.set_num_threads(4)
    t_start = time.time()
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            done = {(r["seed"], r["mode"]) for r in map(json.loads, f)}
    data = CharData()
    with open(OUT, "a") as out:
        for seed in SEEDS:
            for mode in ["clean", "clean+guard", "faults", "faults+guard"]:
                if (seed, mode) in done:
                    continue
                if time.time() - t_start > max_seconds:
                    print("time budget reached; rerun to resume")
                    return
                rec = run(data, seed, mode)
                print(json.dumps(rec))
                out.write(json.dumps(rec) + "\n")
                out.flush()


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1e9)
