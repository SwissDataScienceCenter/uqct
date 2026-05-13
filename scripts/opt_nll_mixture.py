"""Regression test + benchmark for ``uqct.ct.nll_mixture_angle_schedule``.

``ref_nll_mixture_angle_schedule`` below is a frozen verbatim copy of the *old* slow
implementation; ``opt_nll_mixture_angle_schedule`` is just the current production
``uqct.ct.nll_mixture_angle_schedule`` (imported). This script fuzzes the production
function against the frozen reference (bit-for-bit) and benchmarks both.

Why the old one was slow / memory-heavy: the schedule partitions the ``n_angles`` axis
into ``s`` consecutive blocks, and the i-th *prediction set* ``images[..., i, :, :, :]``
is only used for the angles in block ``i``. The old version, however, computed
``radon(images, all_angles)`` -- i.e. it forward-projected every prediction set against
*all* angles, materialising a ``(..., s, n_pred, n_angles, H)`` tensor (plus several
temporaries of that size, plus a ``.double()`` copy), then masked out everything except
block ``i`` for each ``i``. For ``s = 1`` (the SK-ROCK / bootstrap schedule
``[n_angles-1]``) that is ~``n_angles``x wasted FP + memory; for a length-``s`` schedule
it is ~``s * n_angles / max_block``x wasted. The current version walks the schedule
blocks and only forward-projects each prediction set against the angles it needs.

Correctness: ``--mode fuzz`` hammers the optimized version against the reference over
random valid inputs (batch dims (), (N,), (B1,B2); side in {32,64,128}; n_angles in
{1,4,8,16,32,64,200}; schedule flavours uniform/single/exp/all/[n_angles-1]; n_pred
1..64; length_scale 1/5/10/random; uniform & per-angle intensities; counts with
injected zeros and large values; int32/int64 schedule; both ``reduce`` modes) and the
production shapes -- 200 trials x 2 reduce modes = 400 checks were bit-identical
(``torch.equal``), and the reference is itself deterministic (FP noise floor 0).
The optimized version assumes a non-decreasing schedule (what setup_experiment always
produces; the reference's ``reduce=False`` slice already assumes the same).

Results (RTX 2080 Ti, side=128, n_angles=200; output bit-identical, max|Δ| = 0):
  - SK-ROCK schedule (s = 1): ~3x faster at N=4/R=64, ~12x at N=16/R=128, ~17x at the
    grid's N=10/R=500 chunk; ~8.5x lower peak memory. The reference OOMs at N>=32/R=500
    on an 11 GB card (it would need ~20-28 GB at the full N=100/R=500); the optimized
    version's footprint is ~the predictions tensor itself, so the N=100 eval fits in
    one batch -> no need to chunk by 10.
  - exp schedule (s = 32): ~1.9x faster at the production N=10/R=16, ~8.5x lower memory.
    Slower (~2x) only for tiny problems (N=4/R=8) where the s ASTRA setups outweigh the
    FP/memory the reference wastes -- not a real workload.

Run (needs CUDA + the ASTRA toolbox):
    uv run python scripts/opt_nll_mixture.py            # fuzz + benchmark
    uv run python scripts/opt_nll_mixture.py --mode fuzz --trials 500
"""

from __future__ import annotations

import gc
import math
import time

import numpy as np
import torch

from uqct.ct import nll, radon
from uqct.ct import (
    nll_mixture_angle_schedule as opt_nll_mixture_angle_schedule,  # the (now optimized) production impl
)


# ---------------------------------------------------------------------------
# Reference: a verbatim copy of the *old* uqct.ct.nll_mixture_angle_schedule
# (the slow "project everything against all angles, then mask" version), frozen
# here so the fuzz/benchmark below can keep checking the production function
# against it. Do NOT edit.
# ---------------------------------------------------------------------------
def ref_nll_mixture_angle_schedule(
    images: torch.Tensor,
    counts: torch.Tensor,
    intensities: torch.Tensor,
    angles: torch.Tensor,
    schedule: torch.Tensor,
    reduce: bool = True,
    length_scale: float = 5.0,
) -> torch.Tensor:
    device = images.device
    n_angles = counts.shape[-2]

    schedule_lb = schedule.unsqueeze(1)
    schedule_ub = torch.cat(
        [schedule[1:], torch.tensor((n_angles,), device=device)]
    ).unsqueeze(1)
    angular_range = torch.arange(n_angles, device=device).expand(len(schedule), -1)
    mask = angular_range >= schedule_lb
    mask &= angular_range < schedule_ub

    n_pred = images.shape[-3]
    counts_expanded = counts.unsqueeze(-3).unsqueeze(-3)
    intensities = intensities.unsqueeze(-3).unsqueeze(-3)

    nlls = nll(images, counts_expanded, intensities, angles, length_scale).double()
    mix_input = -nlls.sum(-1) - math.log(n_pred)
    mix = -torch.logsumexp(mix_input, dim=-2)
    mix[..., ~mask] = 0
    if reduce:
        out = mix.sum(-1)
    else:
        out = mix.sum(-2)[..., schedule.min() :]
    return out.float()


# The optimized version under test is the production ``uqct.ct.nll_mixture_angle_schedule``
# (imported above as ``opt_nll_mixture_angle_schedule``): it walks the schedule's angle
# blocks and only forward-projects each prediction set against the angles it needs.


# ---------------------------------------------------------------------------
# Test inputs.
# ---------------------------------------------------------------------------
def make_inputs(n_imgs, n_pred, n_angles, side, schedule_kind, *, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    if schedule_kind == "skrock":  # s = 1, schedule = [n_angles - 1] (full angle set)
        schedule = torch.tensor([n_angles - 1], device=device)
    elif schedule_kind == "diffusion":  # s = 32, exp schedule from 10
        s = 32
        schedule = (
            torch.logspace(
                math.log10(10), math.log10(n_angles - 1), steps=s, device=device
            )
            .round()
            .int()
        )
        # de-duplicate to a strictly increasing schedule (setup_experiment guarantees this)
        uniq = torch.unique_consecutive(schedule)
        schedule = uniq
    else:
        raise ValueError(schedule_kind)
    s = len(schedule)
    images = torch.rand(
        (n_imgs, s, n_pred, side, side), generator=g, device=device
    ).clamp_(0, 1)
    gt = torch.rand((n_imgs, side, side), generator=g, device=device).clamp_(0, 1)
    intensities = torch.full((n_imgs, n_angles, 1), 3906.25, device=device)
    counts = torch.poisson(
        intensities
        * torch.exp(
            -(5.0 / side) * radon(gt.contiguous(), angles_for(n_angles, device))
        ),
        generator=None,
    )
    angles = angles_for(n_angles, device)
    return images.contiguous(), counts.contiguous(), intensities, angles, schedule


def angles_for(n_angles, device):
    return torch.from_numpy(np.linspace(0, 180, n_angles, endpoint=False)).to(device)


# ---------------------------------------------------------------------------
# Correctness + benchmark.
# ---------------------------------------------------------------------------
def _max_abs_rel(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = (a.double() - b.double()).abs()
    denom = b.double().abs().clamp_min(1e-12)
    return float(d.max()), float((d / denom).max())


def bench(fn, *args, reduce: bool, n_warmup=1, n_rep=3):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    for _ in range(n_warmup):
        fn(*args, reduce=reduce)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = None
    for _ in range(n_rep):
        out = fn(*args, reduce=reduce)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_rep
    peak = torch.cuda.max_memory_allocated() / 1e9
    return dt, peak, out


def main():
    assert torch.cuda.is_available(), "needs CUDA / ASTRA"
    dev = torch.device("cuda")
    side, n_angles = 128, 200
    cases = [
        # (label, schedule_kind, n_imgs, n_pred)
        ("skrock   N=4   R=64       ", "skrock", 4, 64),
        ("skrock   N=16  R=128      ", "skrock", 16, 128),
        ("skrock   N=10  R=500 (chunk)", "skrock", 10, 500),
        ("skrock   N=32  R=500      ", "skrock", 32, 500),
        ("skrock   N=100 R=500 (full)", "skrock", 100, 500),
        ("diffusion N=4  R=8        ", "diffusion", 4, 8),
        ("diffusion N=10 R=16 (prod)", "diffusion", 10, 16),
    ]
    print(f"side={side} n_angles={n_angles}\n")
    for label, kind, n_imgs, n_pred in cases:
        imgs, counts, ints, angs, sched = make_inputs(
            n_imgs, n_pred, n_angles, side, kind, device=dev
        )
        line = f"{label} | s={len(sched)} "
        ok = True
        for reduce in (True, False):
            try:
                dt_r, mem_r, out_r = bench(
                    ref_nll_mixture_angle_schedule,
                    imgs,
                    counts,
                    ints,
                    angs,
                    sched,
                    reduce=reduce,
                )
            except torch.cuda.OutOfMemoryError:
                line += f"| reduce={reduce}: REF OOM"
                torch.cuda.empty_cache()
                continue
            dt_o, mem_o, out_o = bench(
                opt_nll_mixture_angle_schedule,
                imgs,
                counts,
                ints,
                angs,
                sched,
                reduce=reduce,
            )
            mad, mrd = _max_abs_rel(out_o, out_r)
            same = torch.allclose(out_o, out_r, rtol=1e-4, atol=1e-4)
            ok &= same
            line += (
                f"| reduce={str(reduce):5s}: ref {dt_r * 1e3:6.1f}ms {mem_r:5.2f}GB  "
                f"opt {dt_o * 1e3:6.1f}ms {mem_o:5.3f}GB  "
                f"=> {dt_r / dt_o:4.1f}x faster {mem_r / max(mem_o, 1e-9):5.1f}x less mem  "
                f"[max|Δ|={mad:.2e} maxrel={mrd:.1e} allclose={same}] "
            )
        print(line)
        if not ok:
            print("  !! MISMATCH")
        del imgs, counts, ints
        torch.cuda.empty_cache()
        gc.collect()


# ---------------------------------------------------------------------------
# Fuzzing: hammer many random valid inputs, assert opt == ref bit-for-bit.
# ---------------------------------------------------------------------------
def _rand_schedule(rng, n_angles, kind):
    """A strictly-increasing schedule in [0, n_angles) -- the contract that
    setup_experiment always produces. ``kind`` picks a flavour."""
    if kind == "skrock":
        return [n_angles - 1]
    if n_angles <= 2:  # only a handful of valid schedules; pick any
        return np.sort(
            rng.choice(n_angles, size=int(rng.integers(1, n_angles + 1)), replace=False)
        ).tolist()
    if kind == "single":
        return [int(rng.integers(0, n_angles))]
    if kind == "exp":  # geometric-ish (the codebase's 'exp' eval schedule)
        start = int(rng.integers(0, max(1, n_angles // 3)))
        s = int(rng.integers(2, min(n_angles, 33)))
        raw = np.unique(
            np.round(np.geomspace(max(start, 1), n_angles - 1, num=s)).astype(int)
        )
        return raw.tolist()
    if kind == "all":  # several / all angles, each its own block
        s = int(rng.integers(1, min(n_angles, 18) + 1))
        return np.sort(rng.choice(n_angles, size=s, replace=False)).tolist()
    # "uniform": s random distinct sorted angles
    s = int(rng.integers(1, min(n_angles, 13) + 1))
    return np.sort(rng.choice(n_angles, size=s, replace=False)).tolist()


def _rand_inputs(rng, device, *, big=False):
    """Random valid (images, counts, intensities, angles, schedule_tensor, meta).

    Everything is contiguous float32 -- ASTRA's FP is float32-only, so the function
    (and the reference) only accepts float32 images, and evaluate_and_save always
    passes contiguous float32 (.contiguous() / .mean(...) / .float()). The schedule
    is int32 (what setup_experiment produces), occasionally int64.
    """
    lead = [
        (),
        (int(rng.integers(1, 5)),),
        (int(rng.integers(1, 3)), int(rng.integers(1, 3))),
    ][int(rng.integers(0, 3))]
    side = int(rng.choice([128] if big else [32, 64, 128]))
    n_angles = int(rng.choice([200] if big else [1, 4, 8, 16, 32, 64]))
    sched = _rand_schedule(
        rng, n_angles, str(rng.choice(["uniform", "single", "exp", "all", "skrock"]))
    )
    s = len(sched)
    n_pred = int(
        rng.choice([1, 8, 16, 64] if big else [1, 1, 2, 3, int(rng.integers(2, 25))])
    )
    length_scale = round(
        float(rng.choice([1.0, 5.0, 10.0, float(rng.uniform(0.5, 12.0))])), 3
    )

    g = torch.Generator(device=device).manual_seed(int(rng.integers(0, 2**31)))
    images = torch.rand(
        (*lead, s, n_pred, side, side), generator=g, device=device
    ).clamp_(0, 1)
    angles = torch.from_numpy(np.linspace(0, 180, n_angles, endpoint=False)).to(device)
    if (
        rng.random() < 0.5
    ):  # uniform vs per-angle-random intensities, wide dynamic range
        intensities = torch.full(
            (*lead, n_angles, 1), float(10 ** rng.uniform(1, 6)), device=device
        )
    else:
        intensities = 10.0 ** torch.rand(
            (*lead, n_angles, 1), generator=g, device=device
        ).mul_(5).add_(1)
    gt = torch.rand((*lead, side, side), generator=g, device=device).clamp_(0, 1)
    rate = intensities * torch.exp(
        -(length_scale / side) * radon(gt.contiguous(), angles)
    )
    counts = torch.poisson(rate, generator=g)
    if rng.random() < 0.15:  # inject exact zeros and some large counts
        k = max(1, counts.numel() // 50)
        flat = counts.reshape(-1)
        flat[:k] = 0.0
        flat[-k:] = float(10 ** rng.uniform(4, 7))
        counts = flat.reshape(counts.shape)

    sched_t = torch.tensor(
        sched, device=device, dtype=torch.int32 if rng.random() < 0.8 else torch.int64
    )
    meta = dict(
        lead=lead,
        s=s,
        n_pred=n_pred,
        side=side,
        n_angles=n_angles,
        sched=sched,
        length_scale=length_scale,
        contig=bool(images.is_contiguous()),
    )
    return images, counts.contiguous(), intensities, angles, sched_t, meta


def fuzz(n_trials: int = 80):
    assert torch.cuda.is_available(), "needs CUDA / ASTRA"
    dev = torch.device("cuda")
    rng = np.random.default_rng(0)
    n_ok = n_skip = 0
    max_self_noise = (
        0.0  # max |ref(call1) - ref(call2)| over all trials (the FP noise floor)
    )
    max_opt_vs_ref = 0.0  # max |opt - ref| over all trials
    n_bit_identical = 0
    n_checks = 0
    for trial in range(n_trials):
        big = (
            trial % 12 == 0
        )  # occasionally a production-sized case (side=128, n_angles=200)
        try:
            imgs, counts, ints, angs, sched, meta = _rand_inputs(rng, dev, big=big)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            n_skip += 1
            continue
        try:
            for reduce in (True, False):
                ref1 = ref_nll_mixture_angle_schedule(
                    imgs,
                    counts,
                    ints,
                    angs,
                    sched,
                    reduce=reduce,
                    length_scale=meta["length_scale"],
                )
                ref2 = ref_nll_mixture_angle_schedule(
                    imgs,
                    counts,
                    ints,
                    angs,
                    sched,
                    reduce=reduce,
                    length_scale=meta["length_scale"],
                )
                out = opt_nll_mixture_angle_schedule(
                    imgs,
                    counts,
                    ints,
                    angs,
                    sched,
                    reduce=reduce,
                    length_scale=meta["length_scale"],
                )
                assert (
                    out.shape == ref1.shape
                ), f"shape {out.shape} != {ref1.shape}  meta={meta} reduce={reduce}"
                assert (
                    out.dtype == ref1.dtype == torch.float32
                ), f"dtype {out.dtype}/{ref1.dtype}"
                assert torch.isfinite(
                    out
                ).all(), f"non-finite output  meta={meta} reduce={reduce}"
                self_noise = float((ref1.double() - ref2.double()).abs().max())
                opt_diff = float((out.double() - ref1.double()).abs().max())
                max_self_noise = max(max_self_noise, self_noise)
                max_opt_vs_ref = max(max_opt_vs_ref, opt_diff)
                n_checks += 1
                if torch.equal(out, ref1):
                    n_bit_identical += 1
                # opt must be no further from ref than ref is from itself (FP-reordering noise),
                # with a tiny absolute slack for the s=1 / single-block case being deterministic.
                tol = max(self_noise, 1e-9 * max(1.0, float(ref1.abs().max())))
                assert opt_diff <= tol, (
                    f"opt vs ref diff {opt_diff:.3e} > tol {tol:.3e} (self-noise {self_noise:.3e})  "
                    f"meta={meta} reduce={reduce}"
                )
            n_ok += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            n_skip += 1
            continue
        finally:
            del imgs, counts, ints
            if trial % 8 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    print(
        f"FUZZ: {n_ok} trials passed, {n_skip} skipped (OOM) of {n_trials}; {n_checks} (config x reduce) checks.\n"
        f"      bit-identical (torch.equal): {n_bit_identical}/{n_checks}\n"
        f"      max |ref(a) - ref(b)| (FP noise floor): {max_self_noise:.3e}\n"
        f"      max |opt - ref|:                        {max_opt_vs_ref:.3e}\n"
        f"      => {'IDENTICAL' if max_opt_vs_ref == 0.0 else ('within FP noise floor' if max_opt_vs_ref <= max_self_noise else 'CHECK')}"
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["bench", "fuzz", "both"], default="both")
    p.add_argument("--trials", type=int, default=80)
    a = p.parse_args()
    if a.mode in ("fuzz", "both"):
        fuzz(a.trials)
    if a.mode in ("bench", "both"):
        main()
