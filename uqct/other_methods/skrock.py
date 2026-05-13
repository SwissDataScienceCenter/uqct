"""SK-ROCK posterior sampler built on top of ``deepinv.sampling``.

Uses the SK-ROCK kernel (``deepinv.sampling.SKRockIterator``), the sampling
driver (``deepinv.sampling.BaseSampling`` / ``sampling_builder``) and the
log-Poisson data fidelity (``deepinv.optim.data_fidelity.LogPoissonLikelihood``)
shipped by `deepinv <https://deepinv.github.io>`_, so the SK-ROCK math can be
cross-checked against deepinv's reference code.

Beer--Lambert / log-Poisson model
---------------------------------
The codebase's Poisson NLL (``uqct.ct.nll``) is, dropping ``x``-independent terms,

    f(x) = sum_i [ N0 * exp(-mu * (R x)_i) + mu * counts_i * (R x)_i ],
    grad f(x) = mu * R^T ( counts - N0 * exp(-mu * R x) ),

with ``R`` the ASTRA radon operator (``uqct.ct.radon``), ``mu = length_scale / side``
and ``N0 = total_intensity / (n_angles * side)`` the per-detector intensity (i.e.
``experiment.intensities``). deepinv's :class:`~deepinv.optim.data_fidelity.LogPoissonLikelihood`
with parameters ``(N0, mu)`` and a *linear* physics ``A = R`` has distance
``d(z, y) = N0 ( 1^T exp(-mu z) + mu exp(-mu y)^T z )`` with ``z = R x`` and
``grad_d(z, y) = N0 mu ( exp(-mu y) - exp(-mu z) )``. Choosing
``y = sinogram_from_counts(counts, N0) = -log(counts / N0) / mu`` gives
``N0 exp(-mu y) = counts``, hence ``A^T grad_d(A x, y) = mu R^T (counts - N0 exp(-mu R x)) = grad f(x)``.
``uqct.ct.nll`` additionally clips ``counts`` (and ``intensities``) at ``1e-9`` inside the
log; we use the same clip for ``y`` -- a no-op whenever counts are positive. We provide
that closed-form ``grad`` directly (``BeerLambertLikelihood``), so deepinv never
autodiffs through ASTRA.

Forward operator
----------------
``A = uqct.ct.radon``; ``A_adjoint`` is the matched transpose -- exactly ASTRA's
backprojection (``AstraParallelOp3D.adjoint``), which is what ``uqct.ct.make_radon_layer``
registers as the VJP of ``radon`` and therefore what the rest of the codebase already
uses inside ``grad f``. (No autograd / no double-backward needed -- the SK-ROCK kernel
only uses first-order gradients.)

Prior
-----
* ``"tv"`` (default) -- Moreau--Yosida gradient ``(x - prox_{lambda*beta*TV}(x)) / lambda``
  of the TV norm (``1/lambda``-smooth), with Chambolle's prox. ``lambda = my_lambda_factor / L_f``
  (Durmus 2018); if ``tv_weight == -1`` then ``beta = tv_weight_calibration *
  _calibrate_tv_weight(total_intensity)`` -- ``_calibrate_tv_weight`` is the codebase's MAP
  heuristic (``uqct.models.iterative``), and ``tv_weight_calibration`` (default
  ``_DEFAULT_TV_WEIGHT_CALIBRATION = 2.0``) is the ECE-calibration adjustment from
  ``scripts/calibrate_skrock.py`` (the bare heuristic over-covers; ~2x halves the ECE of
  the per-pixel credible intervals). Pass ``tv_weight_calibration=1.0`` for the bare heuristic.
* ``"none"`` -- ``deepinv.optim.ZeroPrior`` (MLE Langevin). The flat-prior posterior does
  not concentrate in the under-determined modes and the chain mixes very slowly; not
  recommended -- kept only for ablations.

Lipschitz / step size
---------------------
On the feasible box ``[0,1]^d`` we have ``exp(-mu R x) <= 1`` (rays sum non-negative
pixels), so ``||grad^2 f|| <= mu^2 N0 ||R||_2^2`` -- a rigorous upper bound (never an
underestimate, unlike a finite-difference Hessian estimate at a noisy point). ``||R||_2^2``
is a fixed geometric constant, estimated by power iteration on ``R^T R``. The SK-ROCK step
size is ``delta = dt_perc * l_s / L_total`` with the stability bound
``l_s = (s - 1/2)^2 (2 - 4 eta / 3) - 3/2`` and ``L_total = L_f`` (no prior) or
``L_f + 1/lambda = 2 L_f`` (Moreau--Yosida TV). The ``[0,1]`` constraint is the
``SKRockIterator`` ``clip=(0,1)`` projection; the FOV mask is applied to the returned
samples.

Output: ``(N, T=1, R, H, W)`` -- ``T = 1`` (full angle set, no schedule), ``R = n_samples``
post-burn-in samples (the replicate axis used by ``evaluate_and_save``).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from deepinv.optim import Prior, ZeroPrior
from deepinv.optim.data_fidelity import LogPoissonLikelihood
from deepinv.physics import LinearPhysics
from deepinv.sampling import sampling_builder

from uqct.ct import (
    AstraParallelOp3D,
    Experiment,
    circular_mask,
    fbp,
    get_astra_geometry_3d,
    radon,
    sinogram_from_counts,
)
from uqct.eval.run import run_evaluation

_LENGTH_SCALE = 5.0
PriorKind = Literal["tv", "none"]


# ---------------------------------------------------------------------------
# Chambolle TV proximal operator (for the Moreau-Yosida envelope of beta * TV).
# ---------------------------------------------------------------------------
def _grad_2d(u: torch.Tensor) -> torch.Tensor:
    """Forward differences with Neumann BC. ``(..., H, W) -> (2, ..., H, W)``."""
    gx = torch.zeros_like(u)
    gy = torch.zeros_like(u)
    gx[..., :, :-1] = u[..., :, 1:] - u[..., :, :-1]
    gy[..., :-1, :] = u[..., 1:, :] - u[..., :-1, :]
    return torch.stack([gx, gy], dim=0)


def _div_2d(p: torch.Tensor) -> torch.Tensor:
    """Discrete divergence: the *negative* adjoint of ``_grad_2d`` (the convention
    Chambolle's prox iteration assumes), i.e. ``<grad u, p> == -<u, div p>``.
    ``p`` has shape ``(2, ..., H, W)``."""
    px, py = p[0], p[1]
    d = torch.zeros_like(px)
    d[..., :, 0] = px[..., :, 0]
    d[..., :, 1:-1] = px[..., :, 1:-1] - px[..., :, :-2]
    d[..., :, -1] = -px[..., :, -2]
    d[..., 0, :] = d[..., 0, :] + py[..., 0, :]
    d[..., 1:-1, :] = d[..., 1:-1, :] + py[..., 1:-1, :] - py[..., :-2, :]
    d[..., -1, :] = d[..., -1, :] - py[..., -2, :]
    return d


def chambolle_tv_prox(
    y: torch.Tensor, lam: float, n_iter: int = 25, tau: float = 0.245
) -> torch.Tensor:
    """Solve ``argmin_x 0.5 ||x - y||^2 + lam * TV(x)`` for batched ``y``.

    Chambolle's (2004) algorithm in explicit projected-gradient form (Euclidean
    projection of ``p`` onto the unit ball). Dual convergence requires ``tau < 1/4``
    in 2D (``||grad^* grad|| <= 8`` so the step must be ``< 2/8``); 0.245 is used.
    Operates on ``(..., H, W)`` tensors; broadcasts over leading dims.
    """
    p = torch.zeros((2,) + y.shape, device=y.device, dtype=y.dtype)
    for _ in range(n_iter):
        upd = _grad_2d(_div_2d(p) - y / lam)
        p = p + tau * upd
        norm = torch.sqrt(p[0] * p[0] + p[1] * p[1]).clamp_min(1.0)
        p = p / norm.unsqueeze(0)
    return y - lam * _div_2d(p)


# ---------------------------------------------------------------------------
# Forward operator: ASTRA radon as a deepinv LinearPhysics.
# ---------------------------------------------------------------------------
class RadonPhysics(LinearPhysics):
    """``A(x) = radon(x)``; ``A_adjoint`` is ASTRA's backprojection (the matched
    transpose used as the VJP of ``radon``).

    Operates on ``(..., H, W)`` images and ``(..., n_angles, H)`` sinograms (no
    channel dimension -- the SK-ROCK kernel and the priors used here are shape-agnostic).
    ASTRA requires contiguous inputs, so we enforce that.
    """

    def __init__(self, angles: torch.Tensor, side: int, device: torch.device):
        super().__init__(img_size=(side, side), device=device)
        self.angles = angles
        self.side = side

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: N802
        return radon(x.contiguous(), self.angles)

    def A_adjoint(self, s: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: N802
        s = s.contiguous()
        *lead, n_angles, n_det = s.shape
        if n_det != self.side:
            raise ValueError(
                f"Sinogram detector dim {n_det} != image side {self.side}."
            )
        s_flat = s.reshape(-1, n_angles, n_det)  # (B, n_angles, side)
        proj_geom, vol_geom = get_astra_geometry_3d(
            self.angles, self.side, s_flat.shape[0]
        )
        op = AstraParallelOp3D(proj_geom, vol_geom)
        vol = op.adjoint(s_flat)  # (B, side, side) -- exactly R^T s
        return vol.reshape(*lead, self.side, self.side)


# ---------------------------------------------------------------------------
# Data fidelity: Beer-Lambert / log-Poisson with a closed-form gradient.
# ---------------------------------------------------------------------------
class BeerLambertLikelihood(LogPoissonLikelihood):
    """``LogPoissonLikelihood`` with the closed-form gradient wired in:
    ``grad_d(z, y) = N0 mu (exp(-mu y) - exp(-mu z))`` and ``grad(x, y, physics) = A^T grad_d(A x, y)``."""

    def grad_d(self, u: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.N0 * self.mu * (torch.exp(-self.mu * y) - torch.exp(-self.mu * u))

    def grad(self, x: torch.Tensor, y: torch.Tensor, physics, *args, **kwargs):
        return physics.A_vjp(x, self.grad_d(physics.A(x), y))


# ---------------------------------------------------------------------------
# Moreau-Yosida TV prior.
# ---------------------------------------------------------------------------
class MoreauYosidaTVPrior(Prior):
    r"""``grad(x) = (x - prox_{lambda * beta * TV}(x)) / lambda`` (a ``1/lambda``-smooth
    approximation to the subgradient of ``beta * TV``)."""

    def __init__(self, beta: float, lam: float, n_inner: int = 25):
        super().__init__()
        self.beta = beta
        self.lam = lam
        self.n_inner = n_inner

    def grad(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        prox = chambolle_tv_prox(x, lam=self.lam * self.beta, n_iter=self.n_inner)
        return (x - prox) / self.lam


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _op_sqnorm(physics: RadonPhysics, n_iter: int = 40) -> float:
    """Power iteration: ``||A||_2^2 = lambda_max(A^T A)``."""
    v = torch.randn(1, physics.side, physics.side, device=physics.angles.device)
    v = v / (v.norm() + 1e-30)
    lam = torch.tensor(1.0)
    for _ in range(n_iter):
        w = physics.A_adjoint(physics.A(v))
        lam = w.norm()
        if lam < 1e-30:
            break
        v = w / lam
    return float(lam.item())


# Calibration of the TV weight beta against the per-pixel credible-interval ECE
# (scripts/calibrate_skrock.py, alpha = 0.05, first 10 lung test images) found the
# bare MAP heuristic below over-covers (cov@95 ~ 0.99); 2 x heuristic minimises ECE
# and brings cov@95 to ~0.93-0.96 across I in {1e6, 1e8, 1e9}. So beta defaults to
# tv_weight_calibration * _calibrate_tv_weight(...); pass tv_weight_calibration=1.0
# (or an explicit tv_weight) for the bare heuristic / paper-spirit value.
_DEFAULT_TV_WEIGHT_CALIBRATION = 2.0


def _calibrate_tv_weight(total_intensity: float, side: int = 128) -> float:
    """Log-linear intensity-aware ``beta`` MAP heuristic (matches ``uqct.models.iterative``,
    area-normalised since SK-ROCK uses sum-TV: ``beta`` ~ 3 at I=1e4, ~30 at I=1e9).
    This is the "paper-spirit" reference value; the production default scales it by
    ``_DEFAULT_TV_WEIGHT_CALIBRATION`` -- see that constant."""
    base = 5.0 * 10.0 ** (4.0 + (math.log10(total_intensity) - 4.0) / 5.0)
    return base / float(side * side)


def _skrock_stability_bound(n_stages: int, eta: float) -> float:
    """``l_s = (s - 1/2)^2 (2 - 4 eta / 3) - 3/2`` (the SK-ROCK step-size bound)."""
    return ((n_stages - 0.5) ** 2) * (2.0 - 4.0 * eta / 3.0) - 1.5


# ---------------------------------------------------------------------------
# Public predictor / runner.
# ---------------------------------------------------------------------------
def get_skrock_predictor(
    n_burnin: int = 1000,
    n_samples: int = 1000,
    n_stages: int = 10,
    eta: float = 0.05,
    dt_perc: float = 0.95,
    prior: PriorKind = "tv",
    tv_weight: float = -1.0,
    tv_weight_calibration: float = _DEFAULT_TV_WEIGHT_CALIBRATION,
    my_lambda_factor: float = 1.0,
    chambolle_iters: int = 25,
    lipschitz_iters: int = 40,
    lipschitz_safety: float = 1.1,
    seed: int | None = 0,
    total_intensity_for_calibration: float | None = None,
    verbose: bool = True,
):
    """Closure factory. The returned ``predictor_fn(experiment, schedule)`` has the
    ``run_evaluation`` signature and returns ``(N, T=1, R, H, W)``.

    ``tv_weight == -1`` (default) auto-sets ``beta = tv_weight_calibration *
    _calibrate_tv_weight(total_intensity)`` (the ECE-calibrated value -- see
    ``_DEFAULT_TV_WEIGHT_CALIBRATION``); pass an explicit ``tv_weight`` to override.
    """

    def predictor_fn(
        experiment: Experiment, schedule: torch.Tensor | None
    ) -> torch.Tensor:
        del schedule  # SK-ROCK uses the full angle set; no schedule.
        if not experiment.sparse:
            raise NotImplementedError("Only the sparse setting is supported.")

        device = experiment.counts.device
        counts = experiment.counts  # (N, n_angles, n_det)
        intensities = experiment.intensities  # (N, n_angles, 1)
        angles = experiment.angles
        n_images, n_angles, side = counts.shape
        mask = circular_mask(side, device=device, dtype=counts.dtype)

        n0_vals = intensities.flatten()
        n0 = float(n0_vals[0].item())
        if not torch.allclose(n0_vals, n0_vals[0], rtol=1e-3):
            raise ValueError("Non-uniform per-angle intensities are not supported.")
        mu = _LENGTH_SCALE / side

        physics = RadonPhysics(angles, side, device)
        data_fidelity = BeerLambertLikelihood(N0=n0, mu=mu)
        # deepinv's log-Poisson convention: y = -log(counts / N0) / mu.
        # sinogram_from_counts divides by length_scale / side == mu, so it returns exactly that.
        y = sinogram_from_counts(counts, intensities).contiguous()

        # Rigorous upper bound on the data-fidelity Lipschitz constant on [0,1]^d.
        sqnorm_r = _op_sqnorm(physics, n_iter=lipschitz_iters)
        lf = max(lipschitz_safety * mu * mu * n0 * sqnorm_r, 1.0)

        if prior == "none":
            prior_obj: Prior = ZeroPrior()
            l_total = lf
            alpha = 0.0
            beta = 0.0
            my_lam = float("nan")
        elif prior == "tv":
            beta = tv_weight
            if beta == -1.0:
                ti = total_intensity_for_calibration
                if ti is None:
                    ti = float(experiment.total_intensity.mean().item())
                beta = tv_weight_calibration * _calibrate_tv_weight(ti, side)
            my_lam = my_lambda_factor / lf
            prior_obj = MoreauYosidaTVPrior(
                beta=beta, lam=my_lam, n_inner=chambolle_iters
            )
            l_total = lf + 1.0 / my_lam  # exact: the MY envelope is (1/lambda)-smooth.
            alpha = 1.0
        else:
            raise ValueError(f"Unknown prior {prior!r}")

        l_s = _skrock_stability_bound(n_stages, eta)
        step_size = dt_perc * l_s / l_total

        if verbose:
            print(
                f"[skrock] N0={n0:.3e} mu={mu:.4e} ||R||^2={sqnorm_r:.3e} "
                f"L_f={lf:.3e} L_total={l_total:.3e}\n"
                f"[skrock] prior={prior} beta={beta:.3e} lambda_MY={my_lam:.3e} "
                f"s={n_stages} eta={eta} l_s={l_s:.2f} dt%={dt_perc} -> delta={step_size:.3e}\n"
                f"[skrock] burnin={n_burnin} samples={n_samples} images={n_images} "
                f"angles={n_angles} side={side}"
            )

        max_iter = n_burnin + n_samples
        sampler = sampling_builder(
            "SKRock",
            data_fidelity=data_fidelity,
            prior=prior_obj,
            params_algo={
                "step_size": step_size,
                "alpha": alpha,
                "inner_iter": n_stages,
                "eta": eta,
                "sigma": 0.0,  # unused by ZeroPrior / MoreauYosidaTVPrior.
            },
            max_iter=max_iter,
            burnin_ratio=n_burnin / max_iter,
            thinning=1,
            history_size=n_samples,
            verbose=verbose,
            clip=(0.0, 1.0),
        )

        x_init = (
            fbp(sinogram_from_counts(counts, intensities).clip(0), angles)
            .clip(0.0, 1.0)
            .contiguous()
        )
        sampler.sample(y, physics, x_init=x_init, seed=seed)
        chain = sampler.get_chain()  # list of dicts, len == n_samples
        if len(chain) != n_samples:
            print(
                f"[skrock] WARNING: kept {len(chain)} samples (expected {n_samples})."
            )
        samples = torch.stack([d["x"] for d in chain], dim=0)  # (R, N, H, W)
        samples = samples.permute(1, 0, 2, 3).contiguous()  # (N, R, H, W)
        samples = (samples * mask).clamp_(0.0, 1.0)
        return samples.unsqueeze(1)  # (N, T=1, R, H, W)

    return predictor_fn


def run_skrock(
    dataset,
    sparse,
    total_intensity,
    image_range,
    seed,
    n_angles,
    max_angle,
    n_burnin: int = 1000,
    n_samples: int = 1000,
    n_stages: int = 10,
    eta: float = 0.05,
    dt_perc: float = 0.95,
    prior: PriorKind = "tv",
    tv_weight: float = -1.0,
    tv_weight_calibration: float = _DEFAULT_TV_WEIGHT_CALIBRATION,
    my_lambda_factor: float = 1.0,
    chambolle_iters: int = 25,
    lipschitz_iters: int = 40,
    sampler_seed: int = 0,
):
    predictor_fn = get_skrock_predictor(
        n_burnin=n_burnin,
        n_samples=n_samples,
        n_stages=n_stages,
        eta=eta,
        dt_perc=dt_perc,
        prior=prior,
        tv_weight=tv_weight,
        tv_weight_calibration=tv_weight_calibration,
        my_lambda_factor=my_lambda_factor,
        chambolle_iters=chambolle_iters,
        lipschitz_iters=lipschitz_iters,
        seed=sampler_seed,
        total_intensity_for_calibration=total_intensity,
    )
    run_evaluation(
        dataset=dataset,
        sparse=sparse,
        total_intensity=total_intensity,
        image_range=image_range,
        seed=seed,
        model_name="skrock",
        predictor_fn=predictor_fn,
        n_angles=n_angles,
        schedule_start=199,
        schedule_type="linear",
        schedule_length=1,
        max_angle=max_angle,
        extra_metadata={
            "n_burnin": n_burnin,
            "n_samples": n_samples,
            "n_stages": n_stages,
            "eta": eta,
            "dt_perc": dt_perc,
            "prior": prior,
            "tv_weight": tv_weight,
            "tv_weight_calibration": tv_weight_calibration,
            "my_lambda_factor": my_lambda_factor,
            "chambolle_iters": chambolle_iters,
            "backend": "deepinv",
        },
    )
