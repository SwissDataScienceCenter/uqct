"""Correctness checks for the SK-ROCK sampler (``uqct.other_methods.skrock``).

The SK-ROCK *kernel* itself is deepinv's (``deepinv.sampling.SKRockIterator``) and is
not re-tested here -- these checks target the codebase-specific glue:

* the TV pieces (``_grad_2d`` / ``_div_2d`` adjointness, the Chambolle prox, the
  Moreau--Yosida TV gradient) -- CPU, synthetic;
* the SK-ROCK step-size bound;
* (CUDA + dataset bundle) that the Beer--Lambert likelihood gradient equals the
  autograd gradient of ``uqct.ct.nll`` -- i.e. the physics matches the rest of the
  project -- and that the optimized ``A_adjoint`` equals the autograd VJP of ``radon``;
* (CUDA + dataset bundle) an end-to-end shape / range smoke test of the predictor.

Run standalone::

    uv run python tests/test_skrock.py

The CUDA-gated checks are skipped (not failed) when no GPU is available.
"""

# ruff: noqa: N802, N803, N806

from __future__ import annotations

import math
import sys
import traceback

import torch

from uqct.other_methods.skrock import (
    _LENGTH_SCALE,
    BeerLambertLikelihood,
    MoreauYosidaTVPrior,
    RadonPhysics,
    _div_2d,
    _grad_2d,
    _op_sqnorm,
    _skrock_stability_bound,
    chambolle_tv_prox,
    get_skrock_predictor,
)

torch.set_default_dtype(torch.float64)
_CUDA = torch.cuda.is_available()


# --------------------------------------------------------------------------- #
# CPU: TV building blocks.
# --------------------------------------------------------------------------- #
def test_grad_div_negative_adjoint() -> None:
    torch.manual_seed(0)
    u = torch.randn(3, 11, 13)
    p = torch.randn(2, 3, 11, 13)
    lhs = (_grad_2d(u) * p).sum()
    rhs = -(u * _div_2d(p)).sum()  # <grad u, p> == -<u, div p>
    assert torch.allclose(lhs, rhs, atol=1e-10, rtol=1e-10), (float(lhs), float(rhs))
    assert _grad_2d(torch.full((5, 5), 0.3)).abs().max() < 1e-12  # kills constants


def test_chambolle_prox_limits_and_optimality() -> None:
    torch.manual_seed(1)
    y = torch.randn(2, 9, 9)
    assert (
        chambolle_tv_prox(y, lam=1e-8, n_iter=50) - y
    ).abs().max() < 1e-5  # ~identity
    x_smooth = chambolle_tv_prox(y, lam=50.0, n_iter=200)
    assert x_smooth.var() < 0.02 * y.var()  # heavy weight -> nearly flat
    for lam in (0.05, 0.5):  # prox should leave the mean unchanged
        assert (
            abs(float(chambolle_tv_prox(y, lam=lam, n_iter=300).mean() - y.mean()))
            < 1e-5
        )

    # x* = prox solves min_x 0.5||x-y||^2 + lam*TV(x): perturbing it can't lower the objective.
    def tv(x):
        dh = x[..., 1:, :] - x[..., :-1, :]
        dw = x[..., :, 1:] - x[..., :, :-1]
        dh = torch.nn.functional.pad(dh, (0, 0, 0, 1))
        dw = torch.nn.functional.pad(dw, (0, 1, 0, 0))
        return torch.sqrt(dh * dh + dw * dw).sum()

    lam = 0.3
    xstar = chambolle_tv_prox(y, lam=lam, n_iter=4000)
    obj0 = 0.5 * ((xstar - y) ** 2).sum() + lam * tv(xstar)
    for _ in range(5):
        d = torch.randn_like(y)
        d = d / d.norm()
        for eps in (1e-3, -1e-3):
            assert (
                float(
                    0.5 * ((xstar + eps * d - y) ** 2).sum() + lam * tv(xstar + eps * d)
                )
                >= float(obj0) - 1e-4
            )


def test_moreau_yosida_tv_grad() -> None:
    # Vanishes on constants (TV(constant) == 0).
    assert (
        MoreauYosidaTVPrior(beta=7.0, lam=0.02, n_inner=400)
        .grad(torch.full((5, 5), 0.4))
        .abs()
        .max()
        < 1e-9
    )

    # grad(x) = (x - prox_{lam*beta*TV}(x)) / lam must be a subgradient of beta*TV at the
    # prox point, i.e. beta*TV(z) >= beta*TV(prox) + <grad, z - prox> for all z (the
    # optimality condition that ties the MY gradient to beta*TV). Pin TV's value too.
    torch.manual_seed(2)
    beta, lam = 2.0, 0.05
    prior = MoreauYosidaTVPrior(beta=beta, lam=lam, n_inner=3000)
    x = torch.rand(1, 8, 8)
    grad = prior.grad(x)
    prox = x - lam * grad

    def tv_iso(z):
        dh = torch.nn.functional.pad(z[..., 1:, :] - z[..., :-1, :], (0, 0, 0, 1))
        dw = torch.nn.functional.pad(z[..., :, 1:] - z[..., :, :-1], (0, 1, 0, 0))
        return torch.sqrt(dh * dh + dw * dw).sum()

    base = beta * tv_iso(prox)
    for z in [
        torch.rand(1, 8, 8),
        torch.zeros(1, 8, 8),
        torch.full((1, 8, 8), 0.5),
        prox + 0.1 * torch.randn(1, 8, 8),
    ]:
        assert (
            float(beta * tv_iso(z) - base - (grad * (z - prox)).sum()) >= -1e-3
        ), float(beta * tv_iso(z))


def test_skrock_stability_bound() -> None:
    # l_s = (s - 1/2)^2 (2 - 4 eta / 3) - 3/2; pin the paper's s=10, eta=0.05 value.
    assert math.isclose(_skrock_stability_bound(10, 0.05), 172.9775, rel_tol=1e-4)
    assert all(_skrock_stability_bound(s, 0.05) > 0 for s in range(2, 30))


# --------------------------------------------------------------------------- #
# CUDA + dataset bundle: physics consistency, adjoint, end-to-end smoke.
# --------------------------------------------------------------------------- #
def _build_experiment(n_images=2, total_intensity=1e8):
    from uqct.eval.run import setup_experiment

    return setup_experiment(
        "lung",
        (0, n_images),
        total_intensity,
        sparse=True,
        seed=0,
        schedule_length=1,
        schedule_start=199,
        schedule_type="linear",
        n_angles=200,
        max_angle=180,
    )


def test_likelihood_grad_matches_nll() -> None:
    if not _CUDA:
        print("  (skipped: no CUDA)")
        return
    from uqct.ct import nll, sinogram_from_counts

    torch.set_default_dtype(torch.float32)
    try:
        gt, exp, _ = _build_experiment()
        n_imgs, _, side = exp.counts.shape
        mu, n0 = _LENGTH_SCALE / side, float(exp.intensities.flatten()[0])
        x = torch.rand(n_imgs, side, side, device=exp.counts.device)

        # reference: autograd of uqct.ct.nll (how MAP / the old SK-ROCK computed grad f)
        xr = x.unsqueeze(1).clone().requires_grad_(True)
        f = nll(
            xr,
            exp.counts.unsqueeze(1),
            exp.intensities.unsqueeze(1),
            exp.angles,
            length_scale=_LENGTH_SCALE,
        ).sum()
        (g_ref,) = torch.autograd.grad(f, xr)
        g_ref = g_ref.detach().squeeze(1)

        phys = RadonPhysics(exp.angles, side, exp.counts.device)
        y = sinogram_from_counts(exp.counts, exp.intensities).contiguous()
        g_new = BeerLambertLikelihood(N0=n0, mu=mu).grad(x, y, phys)
        rel = float((g_new - g_ref).abs().max() / g_ref.abs().max().clamp_min(1e-12))
        assert rel < 1e-4, rel
    finally:
        torch.set_default_dtype(torch.float64)


def test_adjoint_matches_radon_vjp() -> None:
    if not _CUDA:
        print("  (skipped: no CUDA)")
        return
    from uqct.ct import radon

    torch.set_default_dtype(torch.float32)
    try:
        _, exp, _ = _build_experiment()
        n_imgs, n_ang, side = exp.counts.shape
        phys = RadonPhysics(exp.angles, side, exp.counts.device)
        s = torch.randn(n_imgs, n_ang, side, device=exp.counts.device)
        xz = torch.zeros(
            n_imgs, side, side, device=exp.counts.device, requires_grad=True
        )
        (adj_ref,) = torch.autograd.grad(radon(xz, exp.angles), xz, grad_outputs=s)
        assert torch.equal(
            phys.A_adjoint(s), adj_ref.detach()
        )  # exact: same ASTRA BP call
        assert _op_sqnorm(phys, n_iter=20) > 0.0
    finally:
        torch.set_default_dtype(torch.float64)


def test_predictor_smoke() -> None:
    if not _CUDA:
        print("  (skipped: no CUDA)")
        return
    torch.set_default_dtype(torch.float32)
    try:
        gt, exp, sched = _build_experiment(n_images=1)
        out = get_skrock_predictor(
            n_burnin=20, n_samples=10, prior="tv", verbose=False
        )(exp, sched)
        assert out.shape == (1, 1, 10, 128, 128)
        assert (
            torch.isfinite(out).all()
            and float(out.min()) >= 0.0
            and float(out.max()) <= 1.0
        )
    finally:
        torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tests = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    failed = 0
    for t in tests:
        try:
            print(f"{t.__name__} ...", end=" ")
            t()
            print("ok")
        except Exception:  # noqa: BLE001
            failed += 1
            print("FAIL")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
