#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run the simulation study for the adaptive pivot-based shrinkage estimator.

The script generates correlated Poisson-regression data, selects the proposed
estimator's tuning parameters by repeated K-fold cross-validation on one shared
selection dataset, and compares the fitted model with
Lasso, Elastic Net, and MNet on independent test data.  The main quantities of interest
are test Poisson deviance, coefficient mean squared error, and the similarity of
estimated coefficients within strongly correlated signal groups.

Only the convex penalty range, 1 < alpha <= 2, is used in this version.  The ridge
Poisson estimate serves as the pivot.  It is re-estimated from each training sample and
within each cross-validation split so that validation and test observations do not
influence model fitting.

The simulation is organized into a small set of experiments.  One varies the amount
of trust placed in the pivot, another perturbs the pivot to check sensitivity, and the
main comparison evaluates the proposed estimator against the three reference methods.
Optional sample-size experiments can also be enabled through ``CFG.run_axes``.  For
each experiment, the script saves replication-level statistics, summary tables,
figures, confidence intervals, and paired significance tests.

The default settings are collected in ``ExperimentConfig`` below.  For example,
``CFG.scenario_selector = "D"`` runs Scenario D, whereas ``"all"`` runs every
scenario.  Set ``CFG.fast_mode = True`` for a short diagnostic run before starting a
full simulation.  The random seeds are derived from named pieces of the experiment,
which makes the results reproducible even when replications are run in parallel.

Author: Ibrahim Bakari


USAGE:
    CFG.scenario_selector = "B"          # single scenario
    CFG.scenario_selector = "all"        # all four scenarios
    CFG.kappa_grid = (0.08, 0.12)
    CFG.fast_mode  = True                # quick test run
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, wilcoxon
import warnings
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import multiprocessing
import hashlib

try:
    from joblib import Parallel, delayed, parallel_config
    USE_JOBLIB = True
except ImportError:
    USE_JOBLIB = False

try:
    from numba import njit
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False

warnings.filterwarnings('ignore', category=RuntimeWarning)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'figure.titlesize': 14, 'legend.fontsize': 9, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.format': 'pdf'
})


#  SCENARIO DEFINITIONS

@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    n_samples: int
    p_groups: Tuple[int, int, int, int]
    rho_within: Tuple[float, float, float]
    rho_between: float
    rho_noise: float = 0.05

    @property
    def p_total(self) -> int:
        return sum(self.p_groups)

    @property
    def label(self) -> str:
        return f"Scenario {self.name}: n={self.n_samples}, p={self.p_total}, rho_ext={self.rho_between}"


SCENARIOS = {
    "A": ScenarioConfig("A", 140, (10, 10, 10, 30),  (0.9, 0.9, 0.9),  0.1),
    "B": ScenarioConfig("B", 140, (12, 10,  8, 30),  (0.9, 0.85, 0.9), 0.3),
    "C": ScenarioConfig("C", 140, (14, 12, 10, 40),  (0.9, 0.7, 0.9),  0.5),
    "D": ScenarioConfig("D", 120, (20, 20, 20, 100), (0.85, 0.85, 0.9), 0.3),
}


#  EXPERIMENT CONFIGURATION

@dataclass
class ExperimentConfig:
    scenario_selector: Union[str, List[str]] = "D"
    fast_mode: bool = False

    # ---- regime control ----
    convex_only: bool = True

    SELECTED_KAPPA = 0.08  # ← Change to 0.12 when needed
    kappa_grid: Tuple[float, ...] = (SELECTED_KAPPA,)

    # Convex alpha grid: 1 < alpha <= 2
    alpha_grid_full: Tuple[float, ...] = (1.1, 1.25, 1.4, 1.6, 1.8, 2.0)
    d_grid_full: Tuple[float, ...] = (1e-4, 0.3, 0.5, 0.7, 1.0 - 1e-4)

    # numerics
    eps: float = 1e-5
    tol: float = 1e-6
    max_outer_iter: int = 50
    max_cd_sweeps: int = 20

    lambda_grid: Tuple = field(default_factory=lambda: tuple(np.logspace(-2.5, 0.0, 8).tolist()))
    ridge_lambda_grid: Tuple = field(default_factory=lambda: tuple(np.logspace(-2, 1, 6).tolist()))

    # One independent selection dataset is evaluated with repeated K-fold CV.
    n_folds: int = 5
    cv_repeats: int = 10
    n_replications: int = 200

    test_frac: float = 0.30
    test_size: Optional[int] = 500

    # parallelism
    parallel_reps: bool = True
    max_parallel_jobs: Optional[int] = None
    parallel_backend: str = 'loky'
    inner_max_num_threads: int = 1
    parallel_batch_size: Union[int, str] = 'auto'
    include_baselines: bool = True
    coarse_to_fine: bool = True
    aggressive_early_stop: bool = True

    # Axis 3a and Axis 3b are intentionally disabled(Post_run).(85% of run time  )
    # run_axes: Tuple = ('axis1', 'axis2', 'axis3', 'axis3b', coef_path , 'axis4', 'baseline')


    run_axes: Tuple = ('axis1', 'axis2', 'coef_path', 'axis4', 'baseline')

    # Run Axis 3a and Axis 3b only after all main-axis outputs are saved.
    post_run_axes: Tuple = ('axis3', 'axis3b')


    coefficient_path_lambda_grid: Tuple[float, ...] = field(
        default_factory=lambda: tuple(np.logspace(1.0, -3.0, 80).tolist()))
    coefficient_path_zero_tol: float = 1e-4

    axis3_n_values: Optional[Tuple] = None
    # axis3b: fixed p>n / near-n sweep
    axis3b_n_values: Tuple[int, ...] = (50, 70, 80, 100, 120, 150, 160)

    # bootstrap CIs
    n_bootstrap: int = 1000
    bootstrap_alpha: float = 0.05   # → 95 % CI

    output_dir: Path = Path("results_v1_convex_train_test")
    seed_base: int = 1992

    _current_scenario: Optional[ScenarioConfig] = field(default=None, init=False, repr=False)
    _current_kappa: float = field(default=0.12, init=False, repr=False)

    def __post_init__(self):
        if self.fast_mode:
            self.max_outer_iter, self.max_cd_sweeps = 15, 8
            self.n_folds, self.n_replications = 3, 4
            self.cv_repeats = 2
            self.tol = 1e-4
            self.lambda_grid = tuple(np.logspace(-1.5, 0.5, 6).tolist())
            self.ridge_lambda_grid = tuple(np.logspace(-2, 1, 4).tolist())
            if self.axis3_n_values is None:
                self.axis3_n_values = (100, 400)
        else:
            if self.axis3_n_values is None:
                self.axis3_n_values = (300, 400, 600, 800, 1000, 1200)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def alpha_grid(self) -> Tuple[float, ...]:
        if self.convex_only:
            return tuple(a for a in self.alpha_grid_full if a > 1.0)
        return self.alpha_grid_full

    @property
    def d_grid(self) -> Tuple[float, ...]:
        return self.d_grid_full

    @property
    def scenarios_to_run(self) -> List[str]:
        if isinstance(self.scenario_selector, str):
            if self.scenario_selector.lower() == "all":
                return list(SCENARIOS.keys())
            return [self.scenario_selector]
        return list(self.scenario_selector)

    @property
    def current_scenario(self) -> Optional[ScenarioConfig]:
        return self._current_scenario

    def set_scenario(self, name: str):
        self._current_scenario = SCENARIOS[name]

    def set_kappa(self, k: float):
        self._current_kappa = k

    @property
    def current_kappa(self) -> float:
        return self._current_kappa

    def get_coarse_grid(self):
        ca = tuple(np.array(self.alpha_grid)[::2].tolist()) or self.alpha_grid[:1]
        cd = tuple(np.array(self.d_grid)[::2].tolist()) or self.d_grid[:1]
        cl = tuple(np.array(self.lambda_grid)[::2].tolist()) or self.lambda_grid[:1]
        return ca, cd, cl


CFG: Optional[ExperimentConfig] = None


#  REPRODUCIBLE SEEDING

def stable_seed(seed_base: int, *parts) -> int:
    token = "::".join(str(x) for x in (seed_base,) + parts)
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (2**32 - 1)


def _effective_n_jobs(config: ExperimentConfig, n_tasks: int) -> int:

    available = max((multiprocessing.cpu_count() or 1) - 1, 1)
    if config.max_parallel_jobs is not None:
        available = min(available, max(int(config.max_parallel_jobs), 1))
    return max(1, min(available, max(int(n_tasks), 1)))


def _run_parallel(config: ExperimentConfig, worker, task_args):

    tasks = list(task_args)
    if not tasks:
        return []
    if not (config.parallel_reps and USE_JOBLIB) or len(tasks) == 1:
        return [worker(*args) for args in tasks]

    n_jobs = _effective_n_jobs(config, len(tasks))
    with parallel_config(
        backend=config.parallel_backend,
        n_jobs=n_jobs,
        inner_max_num_threads=max(int(config.inner_max_num_threads), 1),
    ):
        return Parallel(batch_size=config.parallel_batch_size)(
            delayed(worker)(*args) for args in tasks
        )


#  NUMBA KERNELS

if USE_NUMBA:
    @njit(cache=True, fastmath=True)
    def _soft_threshold(a, lam):
        if a > lam:  return a - lam
        if a < -lam: return a + lam
        return 0.0

    @njit(cache=True, fastmath=True)
    def _lqa_weight(bj, alpha, eps):
        return 0.5 * alpha * (abs(bj) + eps) ** (alpha - 2.0)

    @njit(cache=True, fastmath=True)
    def _poisson_deviance(y, mu):
        total = 0.0
        for i in range(len(y)):
            mu_i = mu[i] if mu[i] > 1e-10 else 1e-10
            if y[i] > 0:
                total += y[i] * np.log(y[i] / mu_i) - (y[i] - mu_i)
            else:
                total += mu_i
        return 2.0 * total

    @njit(cache=True, fastmath=True)
    def _cd_proposed(X, w, residual, beta, pivot, denom, lam1, lam2,
                     alpha, eps, is_alpha1, max_cd, tol):
        n, p = X.shape
        for _ in range(max_cd):
            max_chg = 0.0
            for j in range(p):
                rjw = 0.0
                for i in range(n):
                    rjw += w[i] * X[i, j] * residual[i]
                rjw += denom[j] * beta[j]
                num = rjw + 2.0 * lam2 * pivot[j]
                den = denom[j] + 2.0 * lam2
                old = beta[j]
                if is_alpha1:
                    beta[j] = _soft_threshold(num, lam1) / den
                else:
                    vj = _lqa_weight(beta[j], alpha, eps)
                    beta[j] = num / (den + 2.0 * lam1 * vj)
                if beta[j] != old:
                    diff = beta[j] - old
                    for i in range(n):
                        residual[i] -= X[i, j] * diff
                    a = abs(diff)
                    if a > max_chg:
                        max_chg = a
            if max_chg < tol:
                break
        return beta

    @njit(cache=True, fastmath=True)
    def _cd_baseline(X, w, residual, beta, denom, lam1, lam2, ada_w,
                     gamma_mcp, penalty_code, max_cd, tol):
        n, p = X.shape
        for _ in range(max_cd):
            max_chg = 0.0
            beta_snap = beta.copy() if penalty_code == 4 else beta
            for j in range(p):
                rjw = 0.0
                for i in range(n):
                    rjw += w[i] * X[i, j] * residual[i]
                rjw += denom[j] * beta[j]
                old = beta[j]
                if penalty_code == 0:
                    beta[j] = _soft_threshold(rjw, lam1) / denom[j]
                elif penalty_code == 1:
                    beta[j] = _soft_threshold(rjw, lam1) / (denom[j] + 2.0 * lam2)
                elif penalty_code == 2:
                    beta[j] = _soft_threshold(rjw, lam1 * ada_w[j]) / denom[j]
                elif penalty_code == 3:
                    beta[j] = _soft_threshold(rjw, lam1 * ada_w[j]) / (denom[j] + 2.0 * lam2)
                elif penalty_code == 4:
                    abs_b = abs(beta_snap[j])
                    if abs_b <= gamma_mcp * lam1:
                        eff_lam = lam1 - abs_b / gamma_mcp
                    else:
                        eff_lam = 0.0
                    beta[j] = _soft_threshold(rjw, eff_lam) / (denom[j] + 2.0 * lam2)
                else:
                    beta[j] = rjw / (denom[j] + 2.0 * lam1)
                if beta[j] != old:
                    diff = beta[j] - old
                    for i in range(n):
                        residual[i] -= X[i, j] * diff
                    a = abs(diff)
                    if a > max_chg:
                        max_chg = a
            if max_chg < tol:
                break
        return beta
else:
    def _soft_threshold(a, lam):
        return np.sign(a) * max(abs(a) - lam, 0.0)

    def _lqa_weight(bj, alpha, eps):
        return 0.5 * alpha * (abs(bj) + eps) ** (alpha - 2.0)

    def _poisson_deviance(y, mu):
        mu = np.maximum(mu, 1e-10)
        terms = np.where(y > 0, y * np.log(y / mu) - (y - mu), mu)
        return 2.0 * np.sum(terms)

    def _cd_proposed(X, w, residual, beta, pivot, denom, lam1, lam2,
                     alpha, eps, is_alpha1, max_cd, tol):
        n, p = X.shape
        for _ in range(max_cd):
            max_chg = 0.0
            for j in range(p):
                rjw = float(np.dot(w * X[:, j], residual)) + denom[j] * beta[j]
                num = rjw + 2.0 * lam2 * pivot[j]
                den = denom[j] + 2.0 * lam2
                old = beta[j]
                if is_alpha1:
                    beta[j] = _soft_threshold(num, lam1) / den
                else:
                    vj = _lqa_weight(beta[j], alpha, eps)
                    beta[j] = num / (den + 2.0 * lam1 * vj)
                if beta[j] != old:
                    diff = beta[j] - old
                    residual -= X[:, j] * diff
                    if abs(diff) > max_chg:
                        max_chg = abs(diff)
            if max_chg < tol:
                break
        return beta

    def _cd_baseline(X, w, residual, beta, denom, lam1, lam2, ada_w,
                     gamma_mcp, penalty_code, max_cd, tol):
        n, p = X.shape
        for _ in range(max_cd):
            max_chg = 0.0
            beta_snap = beta.copy() if penalty_code == 4 else beta
            for j in range(p):
                rjw = float(np.dot(w * X[:, j], residual)) + denom[j] * beta[j]
                old = beta[j]
                if penalty_code == 0:
                    beta[j] = _soft_threshold(rjw, lam1) / denom[j]
                elif penalty_code == 1:
                    beta[j] = _soft_threshold(rjw, lam1) / (denom[j] + 2.0 * lam2)
                elif penalty_code == 2:
                    beta[j] = _soft_threshold(rjw, lam1 * ada_w[j]) / denom[j]
                elif penalty_code == 3:
                    beta[j] = _soft_threshold(rjw, lam1 * ada_w[j]) / (denom[j] + 2.0 * lam2)
                elif penalty_code == 4:
                    abs_b = abs(beta_snap[j])
                    eff_lam = (lam1 - abs_b / gamma_mcp) if abs_b <= gamma_mcp * lam1 else 0.0
                    beta[j] = _soft_threshold(rjw, eff_lam) / (denom[j] + 2.0 * lam2)
                else:
                    beta[j] = rjw / (denom[j] + 2.0 * lam1)
                if beta[j] != old:
                    diff = beta[j] - old
                    residual -= X[:, j] * diff
                    if abs(diff) > max_chg:
                        max_chg = abs(diff)
            if max_chg < tol:
                break
        return beta


def poisson_deviance(y, mu):
    return _poisson_deviance(np.asarray(y, dtype=np.float64),
                              np.asarray(mu, dtype=np.float64))


#  DATA GENERATION

def generate_covariance(p_groups, rho_within, rho_ext, rho_noise=0.05):
    n_signal = len(p_groups) - 1
    p = sum(p_groups)
    Sigma = np.full((p, p), rho_noise)
    starts = np.cumsum([0] + list(p_groups))
    for g in range(n_signal):
        s, e = starts[g], starts[g + 1]
        pg = e - s
        rg = rho_within[g]
        Sigma[s:e, s:e] = (1 - rg) * np.eye(pg) + rg * np.ones((pg, pg))
    for g1 in range(n_signal):
        for g2 in range(n_signal):
            if g1 != g2:
                s1, e1 = starts[g1], starts[g1 + 1]
                s2, e2 = starts[g2], starts[g2 + 1]
                Sigma[s1:e1, s2:e2] = rho_ext
    ns = starts[n_signal]
    pn = p_groups[-1]
    Sigma[ns:, ns:] = (1 - rho_noise) * np.eye(pn) + rho_noise * np.ones((pn, pn))
    w, v = np.linalg.eigh(Sigma)
    w = np.maximum(w, 1e-8)
    Sigma = v @ np.diag(w) @ v.T
    return (Sigma + Sigma.T) / 2.0, starts


def generate_data(n, p_groups, rho_within, rho_ext, rng=None,
                  signal_strength=0.12, rho_noise=0.05):
    if rng is None:
        rng = np.random.default_rng()
    p = sum(p_groups)
    Sigma, starts = generate_covariance(p_groups, rho_within, rho_ext, rho_noise)
    L = np.linalg.cholesky(Sigma)
    X = rng.standard_normal((n, p)) @ L.T
    X = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=0) + 1e-12)
    perm = rng.permutation(p)
    X = X[:, perm]

    n_active = max(int(np.round(signal_strength * p)), 1)
    active_idx = rng.choice(p, size=n_active, replace=False)
    beta_true = np.zeros(p)
    beta_true[active_idx] = rng.choice([-1, 1], size=n_active) * 0.6

    eta = X @ beta_true
    eta = np.clip(eta, -3.5, 3.5)
    y = rng.poisson(np.exp(eta))

    corr_mat = np.corrcoef(X.T)
    inv_perm = np.argsort(perm)
    masks = {}
    n_sig = len(p_groups) - 1
    for g in range(n_sig):
        orig = np.arange(starts[g], starts[g + 1])
        masks[f"g{g+1}"] = inv_perm[orig]
    masks["noise"] = inv_perm[np.arange(starts[n_sig], p)]
    return X, y, beta_true, corr_mat, masks, perm


def _resolve_test_size(n_train, test_frac=0.30, test_size=None):
    if test_size is not None:
        return max(1, int(test_size))
    return max(1, int(np.ceil(float(test_frac) * int(n_train))))


def generate_train_test_data(n_train, p_groups, rho_within, rho_ext, rng=None,
                             signal_strength=0.12, rho_noise=0.05,
                             test_frac=0.30, n_test=None):
    if rng is None:
        rng = np.random.default_rng()
    p = sum(p_groups)
    n_test = _resolve_test_size(n_train, test_frac=test_frac, test_size=n_test)

    Sigma, starts = generate_covariance(p_groups, rho_within, rho_ext, rho_noise)
    L = np.linalg.cholesky(Sigma)
    X_train = rng.standard_normal((n_train, p)) @ L.T
    X_test = rng.standard_normal((n_test, p)) @ L.T

    mean = X_train.mean(axis=0)
    sd = X_train.std(axis=0, ddof=0) + 1e-12
    X_train = (X_train - mean) / sd
    X_test = (X_test - mean) / sd

    perm = rng.permutation(p)
    X_train = X_train[:, perm]
    X_test = X_test[:, perm]

    n_active = max(int(np.round(signal_strength * p)), 1)
    active_idx = rng.choice(p, size=n_active, replace=False)
    beta_true = np.zeros(p)
    beta_true[active_idx] = rng.choice([-1, 1], size=n_active) * 0.6

    eta_train = X_train @ beta_true
    eta_test = X_test @ beta_true
    y_train = rng.poisson(np.exp(np.clip(eta_train, -3.5, 3.5)))
    y_test = rng.poisson(np.exp(np.clip(eta_test, -3.5, 3.5)))

    corr_mat = np.corrcoef(X_train.T)
    inv_perm = np.argsort(perm)
    masks = {}
    n_sig = len(p_groups) - 1
    for g in range(n_sig):
        orig = np.arange(starts[g], starts[g + 1])
        masks[f"g{g+1}"] = inv_perm[orig]
    masks["noise"] = inv_perm[np.arange(starts[n_sig], p)]
    return X_train, y_train, X_test, y_test, beta_true, corr_mat, masks, perm


#  RIDGE PIVOT

def fit_poisson_ridge(X, y, lam0, max_iter=100, tol=1e-7):
    n, p = X.shape
    beta = np.zeros(p)
    I = np.eye(p)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -10, 10)
        mu = np.exp(eta)
        w = np.maximum(mu, 1e-8)
        z = eta + (y - mu) / w
        H = X.T @ (w[:, None] * X) + lam0 * I
        g = X.T @ (w * z)
        try:
            beta_new = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(H, g, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < tol:
            break
        beta = beta_new
    return beta


def cv_ridge_pivot(X, y, lambda_grid, n_folds, rng):
    n = X.shape[0]
    idx = rng.permutation(n)
    fs = n // n_folds
    folds = [idx[k * fs:(k + 1) * fs] for k in range(n_folds)]
    best_dev, best_lam = np.inf, lambda_grid[0]
    for lam in lambda_grid:
        total = 0.0
        for k in range(n_folds):
            vi = folds[k]
            ti = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            bh = fit_poisson_ridge(X[ti], y[ti], lam, max_iter=50)
            mu_v = np.exp(np.clip(X[vi] @ bh, -10, 10))
            # per-obs scale for consistency with main CV
            total += poisson_deviance(y[vi], mu_v) / max(len(vi), 1)
        avg = total / n_folds
        if avg < best_dev:
            best_dev, best_lam = avg, lam
    return fit_poisson_ridge(X, y, best_lam), best_lam


#  PROPOSED ESTIMATOR

def fit_proposed(X, y, lam1, lam2, d, alpha, beta_pivot,
                 tol=1e-6, max_outer=50, max_cd=20, eps=1e-5):
    n, p = X.shape
    pivot = d * beta_pivot
    beta = pivot.copy()
    Xsq = X * X
    is_alpha1 = abs(alpha - 1.0) < 1e-8
    L_old = np.inf

    for _ in range(max_outer):
        eta = np.clip(X @ beta, -10, 10)
        mu = np.exp(eta)
        w = np.maximum(mu, 1e-8)
        z = eta + (y - mu) / w
        denom = (w[:, None] * Xsq).sum(axis=0)
        residual = z - X @ beta

        beta = _cd_proposed(X, w, residual, beta, pivot, denom,
                            lam1, lam2, alpha, eps, is_alpha1, max_cd, 1e-8)

        eta = np.clip(X @ beta, -10, 10)
        mu = np.exp(eta)
        nll = float(np.sum(mu - y * eta))
        pen = lam1 * float(np.sum((np.abs(beta) + eps) ** alpha)) \
              + lam2 * float(np.sum((beta - pivot) ** 2))
        L_new = nll + pen
        if abs(L_new - L_old) / (abs(L_new) + 1e-10) < tol:
            break
        L_old = L_new
    return beta


#  BASELINES

def _fit_baseline(X, y, lam1, lam2, ada_w, gamma_mcp, penalty_code,
                   max_outer=30, max_cd=20, tol=1e-5):
    n, p = X.shape
    beta = np.zeros(p)
    Xsq = X * X
    if ada_w is None:
        ada_w = np.ones(p)
    for _ in range(max_outer):
        eta = np.clip(X @ beta, -10, 10)
        mu = np.exp(eta)
        w = np.maximum(mu, 1e-8)
        z = eta + (y - mu) / w
        denom = (w[:, None] * Xsq).sum(axis=0)
        residual = z - X @ beta
        beta = _cd_baseline(X, w, residual, beta, denom, lam1, lam2,
                            ada_w, gamma_mcp, penalty_code, max_cd, 1e-8)
        eta2 = np.clip(X @ beta, -10, 10)
        mu2 = np.exp(eta2)
        if np.sqrt(float(np.sum((mu2 - mu) ** 2)) / n) < tol:
            break
    return beta


def fit_lasso(X, y, lam1, **kw):
    return _fit_baseline(X, y, lam1, 0.0, None, 3.0, 0)

def fit_enet(X, y, lam1, lam2, **kw):
    return _fit_baseline(X, y, lam1, lam2, None, 3.0, 1)

def fit_mnet(X, y, lam1, lam2, gamma=3.0, **kw):
    return _fit_baseline(X, y, lam1, lam2, None, gamma, 4)


#  CV TUNING
def _make_repeated_cv_splits(n, n_folds, n_repeats, rng):
    """Create deterministic repeated K-fold splits without dropping remainders."""
    n = int(n)
    n_folds = int(n_folds)
    n_repeats = int(n_repeats)
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_folds > n:
        raise ValueError("n_folds cannot exceed the sample size")
    if n_repeats < 1:
        raise ValueError("cv_repeats must be at least 1")

    all_idx = np.arange(n)
    splits = []
    for repeat in range(n_repeats):
        shuffled = rng.permutation(all_idx)
        fold_parts = np.array_split(shuffled, n_folds)
        for fold, valid_idx in enumerate(fold_parts):
            train_idx = np.concatenate(
                [fold_parts[j] for j in range(n_folds) if j != fold]
            )
            splits.append({
                'repeat': repeat,
                'fold': fold,
                'train_idx': np.asarray(train_idx, dtype=int),
                'valid_idx': np.asarray(valid_idx, dtype=int),
            })
    return splits


def _summarize_repeated_cv(fold_table):
    """Summarize fold deviances first within repeats and then across repeats."""
    key_cols = ['alpha', 'd', 'lambda1', 'lambda2']
    repeat_table = (
        fold_table
        .groupby(key_cols + ['repeat'], as_index=False)
        .agg(repeat_deviance=('fold_deviance', 'mean'))
    )
    summary = (
        repeat_table
        .groupby(key_cols, as_index=False)
        .agg(
            cv_deviance_mean=('repeat_deviance', 'mean'),
            cv_deviance_sd=('repeat_deviance', 'std'),
            cv_repeats=('repeat_deviance', 'size'),
        )
    )
    summary['cv_deviance_sd'] = summary['cv_deviance_sd'].fillna(0.0)
    summary['cv_deviance_se'] = (
        summary['cv_deviance_sd'] /
        np.sqrt(np.maximum(summary['cv_repeats'].to_numpy(dtype=float), 1.0))
    )
    return repeat_table, summary


def cv_tune_proposed(X, y, config: ExperimentConfig, rng,
                     force_full_folds: bool = True):
    """Tune ``(alpha, d, lambda1, lambda2)`` using repeated K-fold CV.

    One fixed selection dataset is reused across all repeats.  Every candidate is
    evaluated on every fold.The selected candidate minimizes the mean deviance across repeat-level means,
    with repeated-CV standard deviation used only as a deterministic tie-break.
    """
    n_folds = int(config.n_folds)
    n_repeats = int(config.cv_repeats)
    splits = _make_repeated_cv_splits(len(y), n_folds, n_repeats, rng)

    # Re-estimate the ridge pivot inside every training fold.

    fold_pivots = []
    inner_folds = min(3, max(2, n_folds - 1))
    for split_id, split in enumerate(splits):
        ti = split['train_idx']
        sub_rng = np.random.default_rng(
            stable_seed(config.seed_base, 'repeated_cv_pivot', split_id,
                        split['repeat'], split['fold'])
        )
        pivot, _ = cv_ridge_pivot(
            X[ti], y[ti], config.ridge_lambda_grid, inner_folds, sub_rng
        )
        fold_pivots.append(pivot)

    if config.coarse_to_fine:
        s_alpha, s_d, s_lam = config.get_coarse_grid()
    else:
        s_alpha = config.alpha_grid
        s_d = config.d_grid
        s_lam = config.lambda_grid

    candidates = [
        (float(alpha), float(d), float(lam1), float(lam2))
        for alpha in s_alpha
        for d in s_d
        for lam1 in s_lam
        for lam2 in s_lam
    ]

    def evaluate_candidate(candidate):
        alpha, d, lam1, lam2 = candidate
        rows = []
        for split, pivot in zip(splits, fold_pivots):
            ti = split['train_idx']
            vi = split['valid_idx']
            bh = fit_proposed(
                X[ti], y[ti], lam1, lam2, d, alpha, pivot,
                tol=config.tol,
                max_outer=config.max_outer_iter,
                max_cd=config.max_cd_sweeps,
            )
            mu_v = np.exp(np.clip(X[vi] @ bh, -10, 10))
            fold_dev = poisson_deviance(y[vi], mu_v) / max(len(vi), 1)
            rows.append({
                'repeat': int(split['repeat']),
                'fold': int(split['fold']),
                'alpha': alpha,
                'd': d,
                'lambda1': lam1,
                'lambda2': lam2,
                'fold_deviance': float(fold_dev),
            })
        return rows

    if USE_JOBLIB and config.parallel_reps and len(candidates) > 1:
        n_jobs = _effective_n_jobs(config, len(candidates))
        candidate_rows = Parallel(
            n_jobs=n_jobs, backend='threading', batch_size=1
        )(delayed(evaluate_candidate)(candidate) for candidate in candidates)
    else:
        candidate_rows = [evaluate_candidate(candidate) for candidate in candidates]

    fold_table = pd.DataFrame(
        [row for candidate_result in candidate_rows for row in candidate_result]
    )
    _, candidate_summary = _summarize_repeated_cv(fold_table)
    candidate_summary = candidate_summary.sort_values(
        ['cv_deviance_mean', 'cv_deviance_sd',
         'alpha', 'd', 'lambda1', 'lambda2'],
        ascending=[True, True, True, True, True, True],
    ).reset_index(drop=True)

    best = candidate_summary.iloc[0]
    best_par = (
        float(best['alpha']), float(best['d']),
        float(best['lambda1']), float(best['lambda2'])
    )

    # Fit the final ridge pivot
    final_pivot_rng = np.random.default_rng(
        stable_seed(config.seed_base, 'repeated_cv_final_pivot')
    )
    final_pivot, _ = cv_ridge_pivot(
        X, y, config.ridge_lambda_grid, config.n_folds, final_pivot_rng
    )
    return best_par, final_pivot, fold_table, candidate_summary, float(best['cv_deviance_mean'])


def select_global_optimal_params(scenario, kappa, config: ExperimentConfig):
    """Select parameters once using repeated K-fold CV on one shared dataset."""
    print(
        f"  [Selection] Repeated CV on one training dataset: "
        f"{config.cv_repeats} repeats x {config.n_folds} folds"
    )
    rng = np.random.default_rng(
        stable_seed(config.seed_base, 'selection_data', scenario.name, kappa)
    )
    X_tr, y_tr, _, _, _, _, _, _ = generate_train_test_data(
        scenario.n_samples, scenario.p_groups, scenario.rho_within,
        scenario.rho_between, rng=rng, signal_strength=kappa,
        rho_noise=scenario.rho_noise, test_frac=config.test_frac,
        n_test=config.test_size,
    )
    cv_rng = np.random.default_rng(
        stable_seed(config.seed_base, 'selection_repeated_cv',
                    scenario.name, kappa)
    )
    (alpha, d, lambda1, lambda2), _, fold_table, summary, cv_mean = \
        cv_tune_proposed(X_tr, y_tr, config, cv_rng, force_full_folds=True)

    best = summary.iloc[0]
    params = {
        'alpha': float(alpha),
        'd': float(d),
        'lambda1': float(lambda1),
        'lambda2': float(lambda2),
        'cv_deviance': float(cv_mean),
        'cv_deviance_sd': float(best['cv_deviance_sd']),
        'cv_deviance_se': float(best['cv_deviance_se']),
    }
    print(
        "  Selected by minimum repeated-CV mean deviance: "
        f"alpha={params['alpha']:.4g}, d={params['d']:.4g}, "
        f"lambda1={params['lambda1']:.4g}, "
        f"lambda2={params['lambda2']:.4g}, "
        f"CV dev={params['cv_deviance']:.4f} "
        f"(SD={params['cv_deviance_sd']:.4f}, "
        f"SE={params['cv_deviance_se']:.4f})"
    )
    return params, fold_table, summary


def cv_tune_baseline(X, y, fit_fn, lam_grid, lam2_grid, n_folds, rng,
                      aggressive_early_stop=True, **fit_kw):
    n = X.shape[0]
    idx = rng.permutation(n)
    fs = n // n_folds
    folds = [idx[k * fs:(k + 1) * fs] for k in range(n_folds)]
    best_dev, best = np.inf, (lam_grid[0], lam2_grid[0] if lam2_grid[0] is not None else None)
    patience = 2 if aggressive_early_stop else n_folds
    for l1 in lam_grid:
        for l2 in lam2_grid:
            total, completed, no_imp, best_fold = 0.0, 0, 0, np.inf
            for k in range(n_folds):
                vi, ti = folds[k], np.concatenate(
                    [folds[j] for j in range(n_folds) if j != k])
                if l2 is None:
                    bh = fit_fn(X[ti], y[ti], l1, **fit_kw)
                else:
                    bh = fit_fn(X[ti], y[ti], l1, l2, **fit_kw)
                mu_v = np.exp(np.clip(X[vi] @ bh, -10, 10))

                fd = poisson_deviance(y[vi], mu_v) / max(len(vi), 1)
                total += fd; completed += 1
                if aggressive_early_stop:
                    if fd < best_fold * 0.99:
                        best_fold = fd; no_imp = 0
                    else:
                        no_imp += 1
                    if no_imp >= patience and k >= 2:
                        break
            avg = total / max(completed, 1)
            if avg < best_dev:
                best_dev, best = avg, (l1, l2)
    return best


#  METRICS

def _fdr(beta_hat, beta_true, tol=1e-6):
    true_a = set(np.where(np.abs(beta_true) > tol)[0])
    hat_a = set(np.where(np.abs(beta_hat) > tol)[0])
    fp = len(hat_a - true_a)
    return fp / max(len(hat_a), 1)


def _group_dispersion(beta, corr, masks, mode='within', tol=0.8):
    p = len(beta)
    diffs = []
    col_group = np.full(p, -1, dtype=int)
    for gid, (name, cols) in enumerate(masks.items()):
        if name != "noise":
            col_group[cols] = gid
    for i in range(p):
        for j in range(i + 1, p):
            if abs(corr[i, j]) > tol:
                gi, gj = col_group[i], col_group[j]
                if mode == 'noise':
                    if gi == -1 and gj == -1:
                        diffs.append(abs(beta[i] - beta[j]))
                elif mode == 'within':
                    if gi >= 0 and gi == gj:
                        diffs.append(abs(beta[i] - beta[j]))
                elif mode == 'external':
                    if gi >= 0 and gj >= 0 and gi != gj:
                        diffs.append(abs(beta[i] - beta[j]))
    return float(np.mean(diffs)) if diffs else np.nan


def compute_metrics(beta_hat, beta_true, X_eval, y_eval, corr, masks, n_obs=None):
    if n_obs is None:
        n_obs = len(y_eval)
    mu_hat = np.exp(np.clip(X_eval @ beta_hat, -10, 10))
    dev = poisson_deviance(y_eval, mu_hat)
    return {
        "Deviance_total": dev,
        "Deviance_per_obs": dev / max(int(n_obs), 1),
        "MAE": float(np.mean(np.abs(y_eval - mu_hat))),
        "RMSE": float(np.sqrt(np.mean((y_eval - mu_hat) ** 2))),
        "MSE": float(np.mean((beta_hat - beta_true) ** 2)),
        "Group_Within": _group_dispersion(beta_hat, corr, masks, 'within'),
        "Group_External": _group_dispersion(beta_hat, corr, masks, 'external'),
        "Group_Noise": _group_dispersion(beta_hat, corr, masks, 'noise'),
    }


def bias_variance_decomposition(beta_hats, beta_trues):

    estimates = np.asarray(beta_hats, dtype=float)
    truths = np.asarray(beta_trues, dtype=float)
    if estimates.shape != truths.shape:
        raise ValueError(
            "beta_hats and beta_trues must have identical (replication, coefficient) shape; "
            f"got {estimates.shape} and {truths.shape}"
        )
    if estimates.ndim != 2 or estimates.shape[0] == 0:
        raise ValueError("bias-variance decomposition requires a non-empty 2D array")

    errors = estimates - truths
    mean_error = errors.mean(axis=0)
    bias_sq = float(np.mean(mean_error ** 2))
    variance = float(np.mean(np.var(errors, axis=0, ddof=0)))
    mse = float(np.mean(errors ** 2))

    if not np.isclose(mse, bias_sq + variance, rtol=1e-10, atol=1e-12):
        raise RuntimeError(
            "Bias-variance identity failed: "
            f"MSE={mse:.12g}, Bias^2+Variance={bias_sq + variance:.12g}"
        )
    return bias_sq, variance, mse


# STATISTICAL TESTING & BOOTSTRAP CIs

def bootstrap_ci(values, n_bootstrap=1000, alpha=0.05, rng_seed=42):
    """
    Percentile bootstrap CI for the mean of `values`.

    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    if len(arr) == 1:
        v = float(arr[0])
        return v, v, v
    rng = np.random.default_rng(rng_seed)
    # (n_bootstrap x n) index matrix → gather → row means, all vectorized
    idx = rng.integers(0, len(arr), size=(n_bootstrap, len(arr)))
    boot_means = arr[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


def paired_wilcoxon(proposed_vals, baseline_vals):
    """
    Two-sided paired Wilcoxon signed-rank test of H0: median(proposed - baseline) = 0.
    Returns (statistic, p_value, direction).
    direction: 'Proposed_better' if proposed < baseline (lower-is-better metrics),
               'Baseline_better' otherwise, 'ns' if p >= 0.05.
    """
    a = np.asarray(proposed_vals, dtype=float)
    b = np.asarray(baseline_vals, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 4:
        return np.nan, np.nan, 'insufficient_data'
    diffs = a - b
    if np.all(diffs == 0):
        return 0.0, 1.0, 'no_difference'
    try:
        stat, pval = wilcoxon(diffs, alternative='two-sided')
    except Exception:
        return np.nan, np.nan, 'error'
    if pval < 0.05:
        direction = 'Proposed_better' if np.median(diffs) < 0 else 'Baseline_better'
    else:
        direction = 'ns'
    return float(stat), float(pval), direction


def add_holm_correction(df, family_cols=('metric',), alpha=0.05):
    """Add Holm-adjusted p-values within prespecified comparison families."""
    out = df.copy()
    out['holm_pval'] = np.nan
    out['holm_reject'] = False
    out['holm_direction'] = ''
    valid = out['wilcoxon_pval'].notna()
    if not valid.any():
        return out
    group_key = family_cols[0] if len(family_cols) == 1 else list(family_cols)
    for _, idx in out.loc[valid].groupby(group_key, dropna=False).groups.items():
        p = out.loc[idx, 'wilcoxon_pval'].to_numpy(dtype=float)
        order = np.argsort(p)
        adjusted_sorted = np.maximum.accumulate(
            (len(p) - np.arange(len(p))) * p[order])
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
        adjusted = np.empty_like(adjusted_sorted)
        adjusted[order] = adjusted_sorted
        out.loc[idx, 'holm_pval'] = adjusted
        out.loc[idx, 'holm_reject'] = adjusted < alpha
        for row_idx, reject in zip(idx, adjusted < alpha):
            if reject:
                out.loc[row_idx, 'holm_direction'] = out.loc[row_idx, 'wilcoxon_direction']
            else:
                out.loc[row_idx, 'holm_direction'] = 'ns'
    return out


def compute_significance_table(by_method, metrics, n_bootstrap=1000,
                                bootstrap_alpha=0.05, label_prefix=''):
    """
    Build a DataFrame with:
      - mean ± bootstrap CI for each method × metric
      - Wilcoxon p-value and direction vs Proposed for each baseline × metric

    """
    methods = [m for m in by_method if by_method[m]]
    rows = []

    for metric in metrics:
        prop_vals = [r[metric] for r in by_method.get('Proposed', [])
                     if metric in r and not np.isnan(r[metric])]

        for method in methods:
            vals = [r[metric] for r in by_method[method]
                    if metric in r and not np.isnan(r[metric])]
            mean_v, ci_lo, ci_hi = bootstrap_ci(vals, n_bootstrap=n_bootstrap,
                                                 alpha=bootstrap_alpha)
            row = {
                'label': label_prefix,
                'method': method,
                'metric': metric,
                'mean': mean_v,
                f'CI_{int((1-bootstrap_alpha)*100)}_lo': ci_lo,
                f'CI_{int((1-bootstrap_alpha)*100)}_hi': ci_hi,
                'n_reps': len(vals),
                'wilcoxon_stat': np.nan,
                'wilcoxon_pval': np.nan,
                'wilcoxon_direction': '',
            }
            if method != 'Proposed' and prop_vals:
                stat, pval, direction = paired_wilcoxon(prop_vals, vals)
                row['wilcoxon_stat'] = stat
                row['wilcoxon_pval'] = pval
                row['wilcoxon_direction'] = direction
            rows.append(row)

    return add_holm_correction(pd.DataFrame(rows), family_cols=('metric',))


#  AXIS 1

def _run_rep_axis1(rep, seed_base, scenario, kappa, d_grid_test,
                    fixed_params, config: ExperimentConfig):
    rng = np.random.default_rng(stable_seed(seed_base, 'a1', scenario.name, kappa, rep))
    X_tr, y_tr, X_te, y_te, bt, corr, masks, _ = generate_train_test_data(
        scenario.n_samples, scenario.p_groups, scenario.rho_within,
        scenario.rho_between, rng=rng, signal_strength=kappa,
        rho_noise=scenario.rho_noise, test_frac=config.test_frac,
        n_test=config.test_size)

    pv_rng = np.random.default_rng(stable_seed(seed_base, 'a1_pivot', scenario.name, kappa, rep))
    pivot, _ = cv_ridge_pivot(X_tr, y_tr, config.ridge_lambda_grid, config.n_folds, pv_rng)
    a_opt = fixed_params['alpha']
    l1_opt = fixed_params['lambda1']
    l2_opt = fixed_params['lambda2']

    by_d = {}
    for d_val in d_grid_test:
        bh = fit_proposed(X_tr, y_tr, l1_opt, l2_opt, d_val, a_opt, pivot,
                           tol=config.tol, max_outer=config.max_outer_iter,
                           max_cd=config.max_cd_sweeps)
        by_d[d_val] = compute_metrics(bh, bt, X_te, y_te, corr, masks, len(y_te))
        by_d[d_val]['n_train'] = len(y_tr)
        by_d[d_val]['n_test'] = len(y_te)

    return {"by_d": by_d, "masks": masks}


def run_axis1(scenario, kappa, fixed_params, config):
    d_grid_test = list(config.d_grid)
    by_d = {d: [] for d in d_grid_test}
    masks_ref = None

    print(f"  [Axis 1] Sensitivity to d with fixed alpha/lambda2; reps={config.n_replications}")

    results = _run_parallel(
        config,
        _run_rep_axis1,
        ((r, config.seed_base, scenario, kappa, d_grid_test, fixed_params, config)
         for r in range(config.n_replications)),
    )

    for r in results:
        for d in d_grid_test:
            by_d[d].append(r["by_d"][d])
        if masks_ref is None:
            masks_ref = r["masks"]

    mse_by_d = {d: float(np.nanmean([rec["MSE"] for rec in by_d[d]])) for d in d_grid_test if by_d[d]}
    dev_by_d = {d: float(np.nanmean([rec["Deviance_per_obs"] for rec in by_d[d]])) for d in d_grid_test if by_d[d]}
    grp_by_d = {d: float(np.nanmean([rec["Group_Within"] for rec in by_d[d]])) for d in d_grid_test if by_d[d]}
    mse_min_d = min(mse_by_d, key=mse_by_d.get) if mse_by_d else None
    dev_min_d = min(dev_by_d, key=dev_by_d.get) if dev_by_d else None
    group_min_d = min(grp_by_d, key=grp_by_d.get) if grp_by_d else None

    summary = {}
    for d in d_grid_test:
        if by_d[d]:
            label = f"Proposed(d={d:.3f})"
            summary[label] = {k: float(np.nanmean([rec[k] for rec in by_d[d]]))
                              for k in by_d[d][0]}

    print(f"\n  {'Method':<28} {'Dev/n':>8} {'MSE':>9} {'GroupW':>9}")
    print("  " + "-" * 58)
    for label in sorted(summary):
        r = summary[label]
        print(f"  {label:<28} {r['Deviance_per_obs']:8.4f} {r['MSE']:9.4f} {r['Group_Within']:9.4f}")

    return {
        "by_d": by_d, "summary": summary,
        "alpha_opt": fixed_params['alpha'],
        "d_opt": fixed_params['d'],
        "lambda1_opt": fixed_params['lambda1'],
        "lambda2_opt": fixed_params['lambda2'],
        "mse_min_d": mse_min_d,
        "dev_min_d": dev_min_d,
        "group_min_d": group_min_d,
        "masks": masks_ref,
        # flat dict keyed by d-label for export_axis_stats
        "_by_label": {f"Proposed(d={d:.3f})": by_d[d] for d in d_grid_test if by_d[d]},
    }


# AXIS 2

def _run_rep_axis2(rep, seed_base, scenario, kappa, fixed_params, config):
    rng = np.random.default_rng(stable_seed(seed_base, 'a2', scenario.name, kappa, rep))
    X_tr, y_tr, X_te, y_te, bt, corr, masks, _ = generate_train_test_data(
        scenario.n_samples, scenario.p_groups, scenario.rho_within,
        scenario.rho_between, rng=rng, signal_strength=kappa,
        rho_noise=scenario.rho_noise, test_frac=config.test_frac,
        n_test=config.test_size)
    p = X_tr.shape[1]

    pv_rng = np.random.default_rng(stable_seed(seed_base, 'a2_pivot', scenario.name, kappa, rep))
    ridge_pivot, _ = cv_ridge_pivot(X_tr, y_tr, config.ridge_lambda_grid, config.n_folds, pv_rng)
    a_opt = fixed_params['alpha']
    d_opt = fixed_params['d']
    l1_opt = fixed_params['lambda1']
    l2_opt = fixed_params['lambda2']

    out = {}
    bh_r = fit_proposed(X_tr, y_tr, l1_opt, l2_opt, d_opt, a_opt, ridge_pivot,
                         tol=config.tol, max_outer=config.max_outer_iter,
                         max_cd=config.max_cd_sweeps)
    out["ridge_pivot_reference"] = compute_metrics(
        bh_r, bt, X_te, y_te, corr, masks, len(y_te))

    for sigma in [0.1, 0.5, 1.0]:
        noisy_pivot = ridge_pivot + rng.normal(0, sigma, size=p)
        bh_n = fit_proposed(X_tr, y_tr, l1_opt, l2_opt, d_opt, a_opt, noisy_pivot,
                             tol=config.tol, max_outer=config.max_outer_iter,
                             max_cd=config.max_cd_sweeps)
        out[f"noisy(sigma={sigma})"] = compute_metrics(bh_n, bt, X_te, y_te, corr,
                                                        masks, len(y_te))

    base_rng = np.random.default_rng(stable_seed(seed_base, 'a2_lasso_pivot', scenario.name, kappa, rep))
    bl = cv_tune_baseline(X_tr, y_tr, fit_lasso, config.lambda_grid, [None],
                           config.n_folds, base_rng,
                           aggressive_early_stop=config.aggressive_early_stop)
    lasso_pivot = fit_lasso(X_tr, y_tr, bl[0])
    bh_l = fit_proposed(X_tr, y_tr, l1_opt, l2_opt, d_opt, a_opt, lasso_pivot,
                         tol=config.tol, max_outer=config.max_outer_iter,
                         max_cd=config.max_cd_sweeps)
    out["lasso_pivot"] = compute_metrics(bh_l, bt, X_te, y_te, corr, masks, len(y_te))

    for rec in out.values():
        rec['n_train'] = len(y_tr)
        rec['n_test'] = len(y_te)
    return {"by_pivot": out}


def run_axis2(scenario, kappa, fixed_params, config):
    pivot_labels = ["ridge_pivot_reference", "noisy(sigma=0.1)", "noisy(sigma=0.5)",
                    "noisy(sigma=1.0)", "lasso_pivot"]
    by_pivot = {lbl: [] for lbl in pivot_labels}

    print(f"  [Axis 2] Pivot robustness with fixed deviance-selected parameters; reps={config.n_replications}")

    results = _run_parallel(
        config,
        _run_rep_axis2,
        ((r, config.seed_base, scenario, kappa, fixed_params, config)
         for r in range(config.n_replications)),
    )

    for r in results:
        for lbl in pivot_labels:
            if lbl in r["by_pivot"]:
                by_pivot[lbl].append(r["by_pivot"][lbl])

    summary = {}
    for lbl in pivot_labels:
        if by_pivot[lbl]:
            summary[lbl] = {k: float(np.nanmean([rec[k] for rec in by_pivot[lbl]]))
                            for k in by_pivot[lbl][0]}

    return {
        "by_pivot": by_pivot, "summary": summary,
        "alpha_opt": fixed_params['alpha'], "d_opt": fixed_params['d'],
        "lambda1_opt": fixed_params['lambda1'], "lambda2_opt": fixed_params['lambda2'],
        # alias for export_axis_stats (ridge is the Proposed reference here)
        "_by_label": by_pivot,
    }


#  COEFFICIENT SHRINKAGE PATH
def run_coefficient_path(scenario, kappa, fixed_params, config, setting_dir):
    """Trace coefficient estimates over lambda1 on one reproducible dataset.

    The proposed method keeps alpha, d, and lambda2 fixed at the global
    CV-selected values.  Lasso varies lambda1 only.  ENet and MNet vary lambda1
    while holding lambda2 at the value selected by their usual CV routines on
    this same training dataset; MNet retains gamma=3.  Because alpha > 1 in the
    convex experiment, these are shrinkage paths rather than exact sparsity
    paths.  ``effective nonzero`` counts use the configured numerical threshold.
    """

    print("  [Coefficient path] Fixed-data shrinkage paths; Axis 3a/3b disabled")

    data_rng = np.random.default_rng(stable_seed(
        config.seed_base, 'coef_path_data', scenario.name, kappa))
    X_tr, y_tr, X_te, y_te, beta_true, corr, masks, _ = generate_train_test_data(
        scenario.n_samples, scenario.p_groups, scenario.rho_within,
        scenario.rho_between, rng=data_rng, signal_strength=kappa,
        rho_noise=scenario.rho_noise, test_frac=config.test_frac,
        n_test=config.test_size)

    pivot_rng = np.random.default_rng(stable_seed(
        config.seed_base, 'coef_path_pivot', scenario.name, kappa))
    pivot, pivot_lambda = cv_ridge_pivot(
        X_tr, y_tr, config.ridge_lambda_grid, config.n_folds, pivot_rng)

    L = config.lambda_grid
    lasso_cv = cv_tune_baseline(
        X_tr, y_tr, fit_lasso, L, [None], config.n_folds,
        np.random.default_rng(stable_seed(
            config.seed_base, 'coef_path_lasso_cv', scenario.name, kappa)),
        aggressive_early_stop=config.aggressive_early_stop)
    enet_cv = cv_tune_baseline(
        X_tr, y_tr, fit_enet, L, list(L), config.n_folds,
        np.random.default_rng(stable_seed(
            config.seed_base, 'coef_path_enet_cv', scenario.name, kappa)),
        aggressive_early_stop=config.aggressive_early_stop)
    mnet_cv = cv_tune_baseline(
        X_tr, y_tr, fit_mnet, L, list(L), config.n_folds,
        np.random.default_rng(stable_seed(
            config.seed_base, 'coef_path_mnet_cv', scenario.name, kappa)),
        aggressive_early_stop=config.aggressive_early_stop, gamma=3.0)

    lambda_path = np.asarray(config.coefficient_path_lambda_grid, dtype=float)
    if lambda_path.ndim != 1 or len(lambda_path) < 2 or np.any(lambda_path <= 0):
        raise ValueError("coefficient_path_lambda_grid must contain at least two positive values")
    lambda_path = np.sort(lambda_path)[::-1]

    alpha = fixed_params['alpha']
    d = fixed_params['d']
    proposed_l2 = fixed_params['lambda2']
    enet_l2 = enet_cv[1]
    mnet_l2 = mnet_cv[1]

    path_fitters = {
        'Proposed': lambda lam1: fit_proposed(
            X_tr, y_tr, lam1, proposed_l2, d, alpha, pivot,
            tol=config.tol, max_outer=config.max_outer_iter,
            max_cd=config.max_cd_sweeps),
        'Lasso': lambda lam1: fit_lasso(X_tr, y_tr, lam1),
        'ENet': lambda lam1: fit_enet(X_tr, y_tr, lam1, enet_l2),
        'MNet': lambda lam1: fit_mnet(X_tr, y_tr, lam1, mnet_l2, gamma=3.0),
    }
    selected_lambda1 = {
        'Proposed': float(fixed_params['lambda1']),
        'Lasso': float(lasso_cv[0]),
        'ENet': float(enet_cv[0]),
        'MNet': float(mnet_cv[0]),
    }
    fixed_lambda2 = {
        'Proposed': float(proposed_l2),
        'Lasso': np.nan,
        'ENet': float(enet_l2),
        'MNet': float(mnet_l2),
    }

    paths = {}
    detail_rows = []
    summary_rows = []
    active = np.abs(beta_true) > 0
    zero_tol = float(config.coefficient_path_zero_tol)

    for method, fit_fn in path_fitters.items():
        method_path = []
        for lam1 in lambda_path:
            beta_hat = np.asarray(fit_fn(float(lam1)), dtype=float)
            method_path.append(beta_hat)
            effective_nonzero = int(np.sum(np.abs(beta_hat) > zero_tol))
            summary_rows.append({
                'method': method,
                'lambda1': float(lam1),
                'log10_lambda1': float(np.log10(lam1)),
                'fixed_lambda2': fixed_lambda2[method],
                'selected_lambda1': selected_lambda1[method],
                'effective_nonzero': effective_nonzero,
                'zero_threshold': zero_tol,
                'coefficient_MSE': float(np.mean((beta_hat - beta_true) ** 2)),
                'max_abs_coefficient': float(np.max(np.abs(beta_hat))),
            })
            for j, (estimate, truth) in enumerate(zip(beta_hat, beta_true)):
                detail_rows.append({
                    'method': method,
                    'lambda1': float(lam1),
                    'log10_lambda1': float(np.log10(lam1)),
                    'coefficient_index': int(j),
                    'estimate': float(estimate),
                    'true_beta': float(truth),
                    'is_active': bool(active[j]),
                })
        paths[method] = np.asarray(method_path)
        print(f"    completed: {method} ({len(lambda_path)} lambda values)")

    detail_path = setting_dir / 'coefficient_path_detail.csv'
    summary_path = setting_dir / 'coefficient_path_summary.csv'
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False, float_format='%.8g')
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, float_format='%.8g')
    print(f"    exported: {detail_path}")
    print(f"    exported: {summary_path}")

    figure_path = setting_dir / 'fig_coefficient_paths.pdf'
    plot_coefficient_paths(
        paths, lambda_path, beta_true, selected_lambda1, scenario, kappa,
        fixed_params, zero_tol, figure_path)

    metadata = pd.DataFrame([{
        'scenario': scenario.name,
        'n_train': len(y_tr),
        'n_test': len(y_te),
        'p': X_tr.shape[1],
        'kappa': float(kappa),
        'alpha': float(alpha),
        'd': float(d),
        'proposed_lambda1_selected': float(fixed_params['lambda1']),
        'proposed_lambda2_fixed': float(proposed_l2),
        'ridge_pivot_lambda': float(pivot_lambda),
        'lasso_lambda1_selected': float(lasso_cv[0]),
        'enet_lambda1_selected': float(enet_cv[0]),
        'enet_lambda2_fixed': float(enet_l2),
        'mnet_lambda1_selected': float(mnet_cv[0]),
        'mnet_lambda2_fixed': float(mnet_l2),
        'mnet_gamma_fixed': 3.0,
        'path_points': len(lambda_path),
        'zero_threshold': zero_tol,
        'seed': stable_seed(config.seed_base, 'coef_path_data', scenario.name, kappa),
    }])
    metadata_path = setting_dir / 'coefficient_path_metadata.csv'
    metadata.to_csv(metadata_path, index=False, float_format='%.8g')
    print(f"    exported: {metadata_path}")

    return {
        'paths': paths,
        'lambda_path': lambda_path,
        'beta_true': beta_true,
        'selected_lambda1': selected_lambda1,
        'fixed_lambda2': fixed_lambda2,
        'pivot_lambda': pivot_lambda,
        'detail_path': detail_path,
        'summary_path': summary_path,
        'metadata_path': metadata_path,
        'figure_path': figure_path,
    }


def plot_coefficient_paths(paths, lambda_path, beta_true, selected_lambda1,
                           scenario, kappa, fixed_params, zero_tol, save_path):
    """Plot active and inactive coefficient trajectories for all four methods."""
    methods = ['Proposed', 'Lasso', 'ENet', 'MNet']
    x = np.log10(np.asarray(lambda_path, dtype=float))
    active_idx = np.where(np.abs(beta_true) > 0)[0]
    inactive_idx = np.where(np.abs(beta_true) == 0)[0]
    active_colors = plt.cm.tab20(np.linspace(0, 1, max(len(active_idx), 1)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
    for ax, method in zip(axes.ravel(), methods):
        B = paths[method]
        for j in inactive_idx:
            ax.plot(x, B[:, j], color='0.72', alpha=0.16, linewidth=0.55)
        for color, j in zip(active_colors, active_idx):
            ax.plot(x, B[:, j], color=color, alpha=0.95, linewidth=1.35)
        ax.axhline(0.0, color='black', linewidth=0.65, alpha=0.7)
        ax.axvline(np.log10(selected_lambda1[method]), color='black',
                   linestyle='--', linewidth=1.25,
                   label=r'CV-selected $\lambda_1$')
        ax.plot([], [], color='0.72', linewidth=1.5, label='Inactive truth')
        ax.plot([], [], color=active_colors[0], linewidth=1.8, label='Active truth')
        ax.set_title(method)
        ax.set_xlabel(r'$\log_{10}(\lambda_1)$')
        ax.set_ylabel(r'Estimated coefficient $\hat\beta_j$')
        ax.grid(True, alpha=0.25)

    axes[0, 0].invert_xaxis()
    axes[0, 0].legend(loc='best', frameon=True, fontsize=8)
    fig.suptitle(
        f'Coefficient Shrinkage Paths - Scenario {scenario.name} '
        f'(n={scenario.n_samples}, p={scenario.p_total}, $\\kappa$={kappa:g})\n'
        f"Proposed: $\\alpha$={fixed_params['alpha']:.3g}, "
        f"$d$={fixed_params['d']:.3g}, $\\lambda_2$={fixed_params['lambda2']:.4g}; "
        f'effective-zero threshold={zero_tol:g}',
        y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path)
    plt.close(fig)
    print(f"    saved: {save_path}")


#  AXIS 3
def _run_rep_axis3(rep, seed_base, scenario, kappa, n_obs, fixed_params, config,
                    run_baselines=True):
    rng = np.random.default_rng(stable_seed(seed_base, 'a3', scenario.name, kappa, n_obs, rep))
    X_tr, y_tr, X_te, y_te, bt, corr, masks, _ = generate_train_test_data(
        n_obs, scenario.p_groups, scenario.rho_within, scenario.rho_between,
        rng=rng, signal_strength=kappa, rho_noise=scenario.rho_noise,
        test_frac=config.test_frac, n_test=config.test_size)

    pv_rng = np.random.default_rng(stable_seed(seed_base, 'a3_pivot', scenario.name, kappa, n_obs, rep))
    pivot, _ = cv_ridge_pivot(X_tr, y_tr, config.ridge_lambda_grid, config.n_folds, pv_rng)
    a = fixed_params['alpha']
    d = fixed_params['d']
    l1 = fixed_params['lambda1']
    l2 = fixed_params['lambda2']
    bh_p = fit_proposed(X_tr, y_tr, l1, l2, d, a, pivot,
                         tol=config.tol, max_outer=config.max_outer_iter,
                         max_cd=config.max_cd_sweeps)
    out = {"Proposed": compute_metrics(bh_p, bt, X_te, y_te, corr, masks, len(y_te))}

    if run_baselines and config.include_baselines:
        br = np.random.default_rng(stable_seed(seed_base, 'a3_base', scenario.name, kappa, n_obs, rep))
        L = config.lambda_grid
        bl = cv_tune_baseline(X_tr, y_tr, fit_lasso, L, [None],  config.n_folds, br,
                              aggressive_early_stop=config.aggressive_early_stop)
        be = cv_tune_baseline(X_tr, y_tr, fit_enet,  L, list(L), config.n_folds, br,
                              aggressive_early_stop=config.aggressive_early_stop)
        bm = cv_tune_baseline(X_tr, y_tr, fit_mnet,  L, list(L), config.n_folds, br,
                              aggressive_early_stop=config.aggressive_early_stop, gamma=3.0)
        out["Lasso"] = compute_metrics(fit_lasso(X_tr, y_tr, bl[0]),       bt, X_te, y_te, corr, masks, len(y_te))
        out["ENet"]  = compute_metrics(fit_enet(X_tr, y_tr, be[0], be[1]), bt, X_te, y_te, corr, masks, len(y_te))
        out["MNet"]  = compute_metrics(fit_mnet(X_tr, y_tr, bm[0], bm[1], gamma=3.0), bt, X_te, y_te, corr, masks, len(y_te))
    for rec in out.values():
        rec['n_train'] = len(y_tr)
        rec['n_test'] = len(y_te)
    return out


def run_axis3(scenario, kappa, fixed_params, config):
    n_values = list(config.axis3_n_values)
    methods = ["Proposed"] + (["Lasso", "ENet", "MNet"] if config.include_baselines else [])
    by_n = {n: {m: [] for m in methods} for n in n_values}
    optimal_params_by_n = {n: fixed_params.copy() for n in n_values}

    print(f"  [Axis 3a] Classical-regime sample-size analysis with fixed "
          f"deviance-selected parameters; n_values={n_values}")

    # Submit the complete (n, replication) grid once.  This reuses a single
    # process pool instead of constructing a new pool for every sample size.
    tasks = [
        (r, config.seed_base, scenario, kappa, n_obs, fixed_params, config, True)
        for n_obs in n_values
        for r in range(config.n_replications)
    ]
    for n_obs in n_values:
        print(f"    queued n = {n_obs}")
    rep_results = _run_parallel(config, _run_rep_axis3, tasks)
    for args, out in zip(tasks, rep_results):
        n_obs = args[4]
        for m in methods:
            if m in out:
                by_n[n_obs][m].append(out[m])

    summary = {}
    for n_obs in n_values:
        for m in by_n[n_obs]:
            if by_n[n_obs][m]:
                summary[f"n{n_obs}_{m}"] = {
                    k: float(np.nanmean([r[k] for r in by_n[n_obs][m]]))
                    for k in by_n[n_obs][m][0]
                }

    _by_label = {}
    for n_obs in n_values:
        for m in by_n[n_obs]:
            if by_n[n_obs][m]:
                _by_label[f"n{n_obs}_{m}"] = by_n[n_obs][m]

    return {"by_n": by_n, "summary": summary,
            "optimal_params": optimal_params_by_n,
            "_by_label": _by_label}


#  AXIS 3B: HIGH-DIMENSIONAL FINITE-SAMPLE SWEEP

def run_axis3b(scenario, kappa, fixed_params, config, setting_dir):
    """
    High-dimensional finite-sample sweep over n in
    {50, 70, 80, 100, 120, 150, 160} with p fixed
    at scenario.p_total (160 for Scenario D).  Produces:
      - axis3b_summary.csv   (mean of each metric per n per method)
      - axis3b_detail.csv    (all per-rep records)
      - axis3b_significance.csv (bootstrap CIs + Wilcoxon tests vs Proposed)
      - fig_axis3b_finite_sample.pdf
    """
    n_values = list(config.axis3b_n_values)
    methods = ["Proposed"] + (["Lasso", "ENet", "MNet"] if config.include_baselines else [])
    sig_metrics = ['MSE', 'Group_Within', 'Deviance_per_obs']

    print(f"  [Axis 3b] High-dimensional finite-sample sweep "
          f"(n <= p={scenario.p_total}); "
          f"n_values={n_values}, reps={config.n_replications}")

    by_n = {n: {m: [] for m in methods} for n in n_values}

    tasks = []
    for n_obs in n_values:
        ratio = scenario.p_total / max(n_obs, 1)
        print(f"    queued n = {n_obs}  (p/n = {ratio:.2f})")
        n_folds_safe = min(config.n_folds, max(2, n_obs // 4))
        tasks.extend(
            (r, config.seed_base, scenario, kappa, n_obs,
             fixed_params, config, n_folds_safe)
            for r in range(config.n_replications)
        )

    rep_results = _run_parallel(config, _run_rep_axis3b, tasks)
    for args, out in zip(tasks, rep_results):
        n_obs = args[4]
        for m in methods:
            if m in out:
                by_n[n_obs][m].append(out[m])

    # summary CSV
    summary_rows = []
    for n_obs in n_values:
        for m in methods:
            recs = by_n[n_obs][m]
            if not recs:
                continue
            row = {'n': n_obs, 'p': scenario.p_total,
                   'p_over_n': round(scenario.p_total / max(n_obs, 1), 4),
                   'method': m, 'n_reps': len(recs)}
            for k in recs[0]:
                if not k.startswith('_'):
                    row[f'mean_{k}'] = float(np.nanmean([r[k] for r in recs]))
                    row[f'sd_{k}']   = float(np.nanstd([r[k] for r in recs], ddof=1))
            summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(setting_dir / 'axis3b_summary.csv', index=False, float_format='%.6g')
    print(f"    exported: {setting_dir / 'axis3b_summary.csv'}")

    #  detail CSV (one row per rep per n per method)
    detail_rows = []
    for n_obs in n_values:
        for m in methods:
            for rep_idx, rec in enumerate(by_n[n_obs][m]):
                row = {'n': n_obs, 'p': scenario.p_total,
                       'p_over_n': round(scenario.p_total / max(n_obs, 1), 4),
                       'method': m, 'rep': rep_idx}
                row.update({k: v for k, v in rec.items() if not k.startswith('_')})
                detail_rows.append(row)
    df_detail = pd.DataFrame(detail_rows)
    df_detail.to_csv(setting_dir / 'axis3b_detail.csv', index=False, float_format='%.6g')
    print(f"    exported: {setting_dir / 'axis3b_detail.csv'}")

    #  significance CSV (original 3-metric format, backward-compatible)
    sig_rows = []
    for n_obs in n_values:
        by_m = {m: by_n[n_obs][m] for m in methods}
        df_sig = compute_significance_table(
            by_m, sig_metrics,
            n_bootstrap=config.n_bootstrap,
            bootstrap_alpha=config.bootstrap_alpha,
            label_prefix=f'n{n_obs}'
        )
        df_sig['n'] = n_obs
        df_sig['p'] = scenario.p_total
        df_sig['p_over_n'] = round(scenario.p_total / max(n_obs, 1), 4)
        sig_rows.append(df_sig)
    df_sig_all = pd.concat(sig_rows, ignore_index=True)
    # Conservative panel-wise family: 7 sample sizes x 3 baselines for each
    # headline metric.  Raw p-values remain available beside Holm p-values.
    df_sig_all = add_holm_correction(df_sig_all, family_cols=('metric',))
    df_sig_all.to_csv(setting_dir / 'axis3b_significance.csv', index=False, float_format='%.6g')
    print(f"    exported: {setting_dir / 'axis3b_significance.csv'}")


    _3b_frames = []
    for n_obs in n_values:
        by_m_slice = {m: by_n[n_obs][m] for m in methods if by_n[n_obs][m]}
        if not by_m_slice:
            continue
        _df_slice = export_axis_stats(
            by_m_slice, None, setting_dir, config,
            proposed_key='Proposed',
            extra_label_cols={'n': n_obs,
                              'p': scenario.p_total,
                              'p_over_n': round(scenario.p_total / max(n_obs, 1), 4)})
        if not _df_slice.empty:
            _3b_frames.append(_df_slice)
    if _3b_frames:
        _df3b = pd.concat(_3b_frames, ignore_index=True)
        _p3b = setting_dir / 'axis3b_stats.csv'
        _df3b.to_csv(_p3b, index=False, float_format='%.6g')
        print(f"    exported: {_p3b}")

    #  plot
    _plot_axis3b(by_n, n_values, methods, scenario, kappa, fixed_params,
                 setting_dir / 'fig_axis3b_finite_sample.pdf')

    return {"by_n": by_n, "summary": df_summary}


def _run_rep_axis3b(rep, seed_base, scenario, kappa, n_obs, fixed_params, config,
                     n_folds_safe):
    """Single replication for Axis 3b (mirrors _run_rep_axis3 but uses n_folds_safe)."""
    rng = np.random.default_rng(stable_seed(seed_base, 'a3b', scenario.name, kappa, n_obs, rep))
    X_tr, y_tr, X_te, y_te, bt, corr, masks, _ = generate_train_test_data(
        n_obs, scenario.p_groups, scenario.rho_within, scenario.rho_between,
        rng=rng, signal_strength=kappa, rho_noise=scenario.rho_noise,
        test_frac=config.test_frac, n_test=config.test_size)

    ridge_lam_grid = config.ridge_lambda_grid
    pv_rng = np.random.default_rng(stable_seed(seed_base, 'a3b_pivot', scenario.name, kappa, n_obs, rep))
    pivot, _ = cv_ridge_pivot(X_tr, y_tr, ridge_lam_grid, n_folds_safe, pv_rng)

    a = fixed_params['alpha']
    d = fixed_params['d']
    l1 = fixed_params['lambda1']
    l2 = fixed_params['lambda2']
    bh_p = fit_proposed(X_tr, y_tr, l1, l2, d, a, pivot,
                         tol=config.tol, max_outer=config.max_outer_iter,
                         max_cd=config.max_cd_sweeps)
    out = {"Proposed": compute_metrics(bh_p, bt, X_te, y_te, corr, masks, len(y_te))}

    if config.include_baselines:
        br = np.random.default_rng(stable_seed(seed_base, 'a3b_base', scenario.name, kappa, n_obs, rep))
        L = config.lambda_grid
        bl = cv_tune_baseline(X_tr, y_tr, fit_lasso, L, [None],  n_folds_safe, br,
                              aggressive_early_stop=config.aggressive_early_stop)
        be = cv_tune_baseline(X_tr, y_tr, fit_enet,  L, list(L), n_folds_safe, br,
                              aggressive_early_stop=config.aggressive_early_stop)
        bm = cv_tune_baseline(X_tr, y_tr, fit_mnet,  L, list(L), n_folds_safe, br,
                              aggressive_early_stop=config.aggressive_early_stop, gamma=3.0)
        out["Lasso"] = compute_metrics(fit_lasso(X_tr, y_tr, bl[0]),       bt, X_te, y_te, corr, masks, len(y_te))
        out["ENet"]  = compute_metrics(fit_enet(X_tr, y_tr, be[0], be[1]), bt, X_te, y_te, corr, masks, len(y_te))
        out["MNet"]  = compute_metrics(fit_mnet(X_tr, y_tr, bm[0], bm[1], gamma=3.0), bt, X_te, y_te, corr, masks, len(y_te))

    for rec in out.values():
        rec['n_train'] = len(y_tr)
        rec['n_test']  = len(y_te)
    return out


def _plot_axis3b(by_n, n_values, methods, scenario, kappa, fixed_params, save_path):
    styles = {
        'Proposed': ('#2E86AB', '-',  'o'),
        'Lasso':    ('#A23B72', '--', 's'),
        'ENet':     ('#F18F01', ':',  '^'),
        'MNet':     ('#C73E1D', '-.', 'P'),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_info = [
        ('MSE',              'Coefficient MSE',       '(a)'),
        ('Group_Within',     'Within-group dispersion','(b)'),
        ('Deviance_per_obs', 'Dev / n (test)',         '(c)'),
    ]
    for ax, (metric, ylabel, panel) in zip(axes, metric_info):
        for m in methods:
            color, ls, mk = styles.get(m, ('#888888', '-', 'x'))
            means, sds = [], []
            for n in n_values:
                recs = by_n[n][m]
                vals = [r[metric] for r in recs if metric in r]
                means.append(float(np.nanmean(vals)) if vals else np.nan)
                sds.append(float(np.nanstd(vals, ddof=1) / np.sqrt(max(len(vals), 1))) if vals else np.nan)
            ax.plot(n_values, means, marker=mk, ls=ls, color=color, lw=2,
                    label=m, markersize=6)
            ax.fill_between(n_values,
                            [u - s for u, s in zip(means, sds)],
                            [u + s for u, s in zip(means, sds)],
                            color=color, alpha=0.12)
        ax.set_xlabel('Training sample size $n$', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'{panel} {ylabel}', fontsize=11)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.3, linestyle='--')

        # mark p=n boundary
        p_total = scenario.p_total
        if min(n_values) < p_total < max(n_values):
            ax.axvline(p_total, color='gray', ls=':', lw=1.5,
                       label=f'p={p_total}')

    fig.suptitle(
        f'Axis 3b: Finite Sample Size Analysis '
        f'[{scenario.name}, $\\kappa$={kappa:g}]',
        fontsize=13, fontweight='bold', y=1.02
    )
    _scenario_annotation(axes[0], scenario, kappa, fixed_params)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    saved: {save_path}")


#  AXIS 4
def _run_rep_axis4(rep, seed_base, scenario, kappa, fixed_params, config):
    rng = np.random.default_rng(stable_seed(seed_base, 'a4', scenario.name, kappa, rep))
    X_tr, y_tr, X_te, y_te, bt, corr, masks, _ = generate_train_test_data(
        scenario.n_samples, scenario.p_groups, scenario.rho_within,
        scenario.rho_between, rng=rng, signal_strength=kappa,
        rho_noise=scenario.rho_noise, test_frac=config.test_frac,
        n_test=config.test_size)

    pv_rng = np.random.default_rng(stable_seed(seed_base, 'a4_pivot', scenario.name, kappa, rep))
    pivot, _ = cv_ridge_pivot(X_tr, y_tr, config.ridge_lambda_grid, config.n_folds, pv_rng)
    a = fixed_params['alpha']
    d = fixed_params['d']
    l1 = fixed_params['lambda1']
    l2 = fixed_params['lambda2']

    fits = {
        "Proposed": fit_proposed(X_tr, y_tr, l1, l2, d, a, pivot,
                                   tol=config.tol, max_outer=config.max_outer_iter,
                                   max_cd=config.max_cd_sweeps),
    }

    if config.include_baselines:
        br = np.random.default_rng(stable_seed(seed_base, 'a4_base', scenario.name, kappa, rep))
        L = config.lambda_grid
        bl = cv_tune_baseline(X_tr, y_tr, fit_lasso, L, [None],  config.n_folds, br,
                              aggressive_early_stop=config.aggressive_early_stop)
        be = cv_tune_baseline(X_tr, y_tr, fit_enet,  L, list(L), config.n_folds, br,
                              aggressive_early_stop=config.aggressive_early_stop)
        bm = cv_tune_baseline(X_tr, y_tr, fit_mnet,  L, list(L), config.n_folds, br,
                              aggressive_early_stop=config.aggressive_early_stop, gamma=3.0)
        fits.update({
            "Lasso": fit_lasso(X_tr, y_tr, bl[0]),
            "ENet":  fit_enet(X_tr, y_tr, be[0], be[1]),
            "MNet":  fit_mnet(X_tr, y_tr, bm[0], bm[1], gamma=3.0),
        })

    metrics = {}
    for name, b in fits.items():
        m = compute_metrics(b, bt, X_te, y_te, corr, masks, len(y_te))
        m['n_train'] = len(y_tr)
        m['n_test'] = len(y_te)
        m['_beta_hat'] = b
        m['_corr_mat'] = corr
        m['_beta_true'] = bt
        metrics[name] = m

    return {"metrics": metrics, "fits": fits,
            "beta_true": bt, "X_train": X_tr, "y_train": y_tr,
            "X_test": X_te, "y_test": y_te,
            "corr": corr, "masks": masks}


def run_axis4(scenario, kappa, fixed_params, config):
    methods = ["Proposed"] + (["Lasso", "ENet", "MNet"] if config.include_baselines else [])
    raw = {m: [] for m in methods}
    masks_ref = None

    print(f"  [Axis 4] Grouping/MSE/deviance with fixed deviance-selected parameters; reps={config.n_replications}")

    results = _run_parallel(
        config,
        _run_rep_axis4,
        ((r, config.seed_base, scenario, kappa, fixed_params, config)
         for r in range(config.n_replications)),
    )

    for r in results:
        for m in methods:
            if m in r["metrics"]:
                raw[m].append(r["metrics"][m])
        if masks_ref is None:
            masks_ref = r["masks"]

    summary = {}
    for m in methods:
        if raw[m]:
            keys = [k for k in raw[m][0] if not k.startswith('_')]
            summary[m] = {k: float(np.nanmean([rec[k] for rec in raw[m]])) for k in keys}

    bias_var = {}
    for m in methods:
        recs = [rec for rec in raw[m]
                if '_beta_hat' in rec and '_beta_true' in rec]
        if len(recs) >= 2:
            b2, var, mse_bv = bias_variance_decomposition(
                [rec['_beta_hat'] for rec in recs],
                [rec['_beta_true'] for rec in recs])
            bias_var[m] = {'Bias_Squared': b2, 'Variance': var, 'MSE': mse_bv}

    return {
        "by_method": raw, "summary": summary, "bias_var": bias_var,
        "alpha_opt": fixed_params['alpha'], "d_opt": fixed_params['d'],
        "lambda1_opt": fixed_params['lambda1'], "lambda2_opt": fixed_params['lambda2'],
        "masks": masks_ref,
    }


def export_axis4_significance(by_method, config, setting_dir):
    """
    Compute axis4_significance.csv

    """
    #  original narrow table (backward-compatible)
    sig_metrics = ['MSE', 'Group_Within', 'Deviance_per_obs']
    df_sig = compute_significance_table(
        by_method, sig_metrics,
        n_bootstrap=config.n_bootstrap,
        bootstrap_alpha=config.bootstrap_alpha,
        label_prefix='axis4'
    )
    path_sig = setting_dir / 'axis4_significance.csv'
    df_sig.to_csv(path_sig, index=False, float_format='%.6g')
    print(f"    exported: {path_sig}")

    #  full-metric stats table
    export_axis_stats(by_method, 'axis4_stats.csv', setting_dir, config,
                      proposed_key='Proposed')
    return df_sig


#  PLOTTING
def _scenario_annotation(ax, scenario, kappa, optimal=None, location=None):



    parts = [f"Scenario {scenario.name}: n={scenario.n_samples}, "
             f"p={scenario.p_total}, rho_ext={scenario.rho_between}",
             fr"$\kappa$ = {kappa:g}"]
    if optimal:
        if 'alpha' in optimal:
            parts.append(fr"$\alpha^*$ = {optimal['alpha']:.2f}")
        if 'd' in optimal:
            parts.append(fr"$d^*$ = {optimal['d']:.3f}")
        if 'lambda1' in optimal:
            parts.append(fr"$\lambda_1$ = {optimal['lambda1']:.4g}")
        if 'lambda2' in optimal:
            parts.append(fr"$\lambda_2$ = {optimal['lambda2']:.4g}")
    fig = ax.figure
    fig.text(0.5, -0.015, '   |   '.join(parts), transform=fig.transFigure,
             fontsize=8, ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                       edgecolor='gray', alpha=0.9))


def plot_axis1(by_d, scenario, kappa, optimal_alpha, optimal_d,
               lambda1_opt, lambda2_opt, save_path):
    d_vals = sorted(by_d.keys())
    mse = [float(np.nanmean([r["MSE"] for r in by_d[d]])) for d in d_vals]
    dev = [float(np.nanmean([r["Deviance_per_obs"] for r in by_d[d]])) for d in d_vals]
    group = [float(np.nanmean([r["Group_Within"] for r in by_d[d]])) for d in d_vals]
    mse_se = [float(np.nanstd([r["MSE"] for r in by_d[d]], ddof=1) /
              np.sqrt(max(len(by_d[d]), 1))) for d in d_vals]
    dev_se = [float(np.nanstd([r["Deviance_per_obs"] for r in by_d[d]], ddof=1) /
              np.sqrt(max(len(by_d[d]), 1))) for d in d_vals]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    cM, cD, cG = '#2E86AB', '#A23B72', '#F18F01'
    ax1.plot(d_vals, mse, 'o-', color=cM, lw=2.5, label='MSE', markersize=7)
    ax1.fill_between(d_vals, [m - s for m, s in zip(mse, mse_se)],
                     [m + s for m, s in zip(mse, mse_se)], color=cM, alpha=0.15)
    ax1.plot(d_vals, group, '^-', color=cG, lw=2.0, label='Group within', markersize=6)
    ax1.set_xlabel('Pivot scaling parameter $d$', fontsize=11)
    ax1.set_ylabel('Coefficient MSE / Group dispersion', color=cM, fontsize=11)
    ax1.tick_params(axis='y', labelcolor=cM)
    ax1.grid(alpha=0.3, linestyle='--')

    ax2 = ax1.twinx()
    ax2.plot(d_vals, dev, 's--', color=cD, lw=2.5, label='Dev/n', markersize=7)
    ax2.fill_between(d_vals, [m - s for m, s in zip(dev, dev_se)],
                     [m + s for m, s in zip(dev, dev_se)], color=cD, alpha=0.15)
    ax2.set_ylabel('Deviance per observation', color=cD, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=cD)

    if optimal_d in d_vals:
        ax1.axvline(optimal_d, color='gray', ls='--', lw=1.5, alpha=0.8,
                    label=fr'$d^*$ selected by CV dev = {optimal_d:.3f}')

    txt = {'alpha': optimal_alpha, 'd': optimal_d,
           'lambda1': lambda1_opt, 'lambda2': lambda2_opt}
    _scenario_annotation(ax1, scenario, kappa, txt, 'upper left')
    plt.suptitle(f'Axis 1: Sensitivity to Pivot Scaling $d$ '
                 f'[{scenario.name}, $\\kappa$={kappa:g}]',
                 fontsize=13, fontweight='bold', y=1.04)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=4, frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    saved: {save_path}")


def plot_axis2(by_pivot, scenario, kappa, optimal_alpha, optimal_d,
               lambda1_opt, lambda2_opt, save_path):
    metrics = ['MSE', 'Group_Within', 'Deviance_per_obs']
    labels = ['MSE', 'Group within', 'Dev / n']
    if not by_pivot.get("ridge_pivot_reference"):
        return
    ref = {m: float(np.nanmean([r[m] for r in by_pivot["ridge_pivot_reference"]]))
           for m in metrics}

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(metrics))
    width = 0.18
    styles = {
        'ridge_pivot_reference': ('#2E86AB', 'Ridge pivot (reference)'),
        'noisy(sigma=0.1)': ('#A23B72', 'Noisy sigma=0.1'),
        'noisy(sigma=0.5)': ('#F18F01', 'Noisy sigma=0.5'),
        'noisy(sigma=1.0)': ('#C73E1D', 'Noisy sigma=1.0'),
        'lasso_pivot':      ('#6A4C93', 'Lasso pivot'),
    }
    for idx, (key, (color, label)) in enumerate(styles.items()):
        if key not in by_pivot or not by_pivot[key]:
            continue
        ratios = [float(np.nanmean([r[m] for r in by_pivot[key]])) / ref[m]
                  if ref[m] > 1e-10 else 1.0 for m in metrics]
        ax.bar(x_pos + idx * width, ratios, width, color=color, label=label, alpha=0.85)

    ax.axhline(1.0, color='gray', ls='--', lw=1, label='Ridge ratio=1')
    ax.set_xlabel('Objective metric', fontsize=11)
    ax.set_ylabel('Ratio relative to ridge pivot (lower = better)', fontsize=11)
    ax.set_xticks(x_pos + 2 * width)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(f'Axis 2: Pivot Robustness for MSE, Grouping and Deviance '
                 f'[{scenario.name}, $\\kappa$={kappa:g}]',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    _scenario_annotation(
        ax, scenario, kappa,
        {'alpha': optimal_alpha, 'd': optimal_d,
         'lambda1': lambda1_opt, 'lambda2': lambda2_opt},
        'upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    saved: {save_path}")


def plot_axis3(by_n, scenario, kappa, optimal_params_by_n, save_path):
    ns = sorted(by_n.keys())
    methods = ['Proposed', 'Lasso', 'ENet', 'MNet']
    styles = {
        'Proposed': ('#2E86AB', '-',  'o'),
        'Lasso':    ('#A23B72', '--', 's'),
        'ENet':     ('#F18F01', ':',  '^'),
        'MNet':     ('#C73E1D', '-.', 'P'),
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in methods:
        color, ls, mk = styles[m]
        mse, mse_se = [], []
        for n in ns:
            if m in by_n[n] and by_n[n][m]:
                vals = [r["MSE"] for r in by_n[n][m]]
                mse.append(float(np.nanmean(vals)))
                mse_se.append(float(np.nanstd(vals, ddof=1) / np.sqrt(max(len(vals), 1))))
            else:
                mse.append(np.nan); mse_se.append(np.nan)
        ax.plot(ns, mse, marker=mk, ls=ls, color=color, lw=2, label=m, markersize=6)
        ax.fill_between(ns, [m_ - s for m_, s in zip(mse, mse_se)],
                        [m_ + s for m_, s in zip(mse, mse_se)], color=color, alpha=0.1)

    ax.set_xlabel('Sample size $n$', fontsize=11)
    ax.set_ylabel('Coefficient MSE', fontsize=11)
    ax.set_title(f'Axis 3a: Classical-Regime Sample-Size Analysis '
                 f'(n > p={scenario.p_total}) '
                 f'[{scenario.name}, $\\kappa$={kappa:g}]',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xscale('log')

    largest_n = ns[-1] if ns else None
    opt = optimal_params_by_n.get(largest_n, {}) if largest_n else {}
    _scenario_annotation(ax, scenario, kappa, opt, 'lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    saved: {save_path}")


def plot_axis4(by_method, masks, bias_var, scenario, kappa,
                optimal_alpha, optimal_d, lambda1_opt, lambda2_opt,
                save_path):
    methods = [m for m in ['Proposed', 'Lasso', 'ENet', 'MNet'] if m in by_method and by_method[m]]
    colors = {
        'Proposed': '#2E86AB',
        'Lasso':    '#A23B72',
        'ENet':     '#F18F01',
        'MNet':     '#C73E1D',
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Axis 4: Grouping, MSE, Deviance & Bias-Variance '
                 f'[{scenario.name}, $\\kappa$={kappa:g}]',
                 fontsize=14, fontweight='bold', y=0.995)
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # (a) Within-group dispersion per signal group
    signal_groups = [k for k in masks.keys() if k != 'noise']
    x = np.arange(len(signal_groups))
    width = 0.8 / max(len(methods), 1)
    for drawn, m in enumerate(methods):
        per_group = []
        for gname in signal_groups:
            cols = list(masks[gname])
            disps_per_rep = []
            for r in by_method[m]:
                if '_beta_hat' in r and '_corr_mat' in r:
                    bh = r['_beta_hat']
                    corr = r['_corr_mat']
                    pairs = []
                    for ii in range(len(cols)):
                        for jj in range(ii + 1, len(cols)):
                            i, j = cols[ii], cols[jj]
                            if abs(corr[i, j]) > 0.8:
                                pairs.append(abs(bh[i] - bh[j]))
                    disps_per_rep.append(float(np.mean(pairs)) if pairs else np.nan)
            per_group.append(float(np.nanmean(disps_per_rep)) if disps_per_rep else np.nan)
        ax1.bar(x + drawn * width - 0.4 + width / 2, per_group, width,
                label=m, color=colors.get(m), alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'G{i+1}' for i in range(len(signal_groups))], fontsize=10)
    ax1.set_ylabel('Within-group dispersion (lower = better)', fontsize=10)
    ax1.set_title('(a) Grouping effect: coefficient similarity', fontsize=11)
    ax1.legend(fontsize=7, frameon=False, ncol=2)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # (b) Objective metrics: MSE, grouping, deviance
    metric_names = ['MSE', 'Group_Within', 'Deviance_per_obs']
    metric_labels = ['MSE', 'Group within', 'Dev / n']
    vals = {mt: np.array([float(np.nanmean([r[mt] for r in by_method[m]])) for m in methods])
            for mt in metric_names}
    norm_vals = {}
    for mt in metric_names:
        arr = vals[mt]
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        norm_vals[mt] = (arr - mn) / (mx - mn) if mx > mn + 1e-12 else np.zeros_like(arr)
    x2 = np.arange(len(metric_names))
    w2 = 0.8 / max(len(methods), 1)
    for idx, m in enumerate(methods):
        ax2.bar(x2 + idx * w2 - 0.4 + w2 / 2,
                [norm_vals[mt][idx] for mt in metric_names], w2,
                label=m, color=colors.get(m), alpha=0.85)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metric_labels, fontsize=9)
    ax2.set_ylabel('Normalized value (lower = better)', fontsize=10)
    ax2.set_title('(b) Objective metrics only', fontsize=11)
    ax2.legend(fontsize=7, frameon=False, ncol=2)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # (c) Bias-variance decomposition
    if bias_var:
        x3 = np.arange(len(methods))
        w3 = 0.35
        b2 = [bias_var.get(m, {}).get('Bias_Squared', np.nan) for m in methods]
        vr = [bias_var.get(m, {}).get('Variance', np.nan) for m in methods]
        ax3.bar(x3 - w3 / 2, b2, w3, label='Bias$^2$', color='#4C72B0', alpha=0.85)
        ax3.bar(x3 + w3 / 2, vr, w3, label='Variance', color='#A23B72', alpha=0.85)
        ax3.set_xticks(x3)
        ax3.set_xticklabels(methods, fontsize=9)
        ax3.set_ylabel('Value (lower = better)', fontsize=10)
        ax3.set_title('(c) Bias-variance decomposition', fontsize=11)
        ax3.legend(fontsize=7, frameon=False)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        ax3.set_axisbelow(True)
    else:
        ax3.axis('off')
        ax3.text(0.5, 0.5, 'bias-variance decomposition unavailable',
                 transform=ax3.transAxes, ha='center', va='center',
                 fontsize=10, color='gray')

    # (d) MSE-Deviance relationship
    for m in methods:
        dev_v = [r['Deviance_per_obs'] for r in by_method[m] if 'Deviance_per_obs' in r]
        mse_v = [r['MSE'] for r in by_method[m] if 'MSE' in r]
        if dev_v and mse_v:
            ax4.scatter(dev_v, mse_v, c=colors.get(m), label=m, alpha=0.6,
                        s=40, edgecolors='white')
    ax4.set_xlabel('Deviance per observation', fontsize=10)
    ax4.set_ylabel('Coefficient MSE', fontsize=10)
    ax4.set_title('(d) MSE-deviance trade-off', fontsize=11)
    ax4.legend(fontsize=7, frameon=False, ncol=2)
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.set_axisbelow(True)

    _scenario_annotation(
        ax1, scenario, kappa,
        {'alpha': optimal_alpha, 'd': optimal_d,
         'lambda1': lambda1_opt, 'lambda2': lambda2_opt})
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    saved: {save_path}")


def plot_baseline_comparison(baselines, scenario, kappa, optimal_alpha,
                               optimal_d, lambda1_opt, lambda2_opt,
                               save_path):
    metrics = ['MSE', 'Group_Within', 'Deviance_per_obs']
    metric_labels = ['MSE', 'Group within', 'Dev / n']
    methods = list(baselines.keys())

    norm_vals = {}
    for m in metrics:
        vals = np.array([float(np.nanmean([r[m] for r in baselines[mt]]))
                         if baselines[mt] else np.nan for mt in methods])
        mn, mx = np.nanmin(vals), np.nanmax(vals)
        if mx > mn + 1e-10:
            norm_vals[m] = [(v - mn) / (mx - mn) if not np.isnan(v) else 0.5
                             for v in vals]
        else:
            norm_vals[m] = [0.5] * len(methods)

    x = np.arange(len(metric_labels))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = plt.cm.tab10
    for idx, mt in enumerate(methods):
        vals = [norm_vals[m][idx] for m in metrics]
        bars = ax.bar(x + idx * width - 0.4 + width / 2, vals, width,
                       label=mt, color=cmap(idx % 10), alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Objective metric', fontsize=11)
    ax.set_ylabel('Normalized value (lower = better)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_title(f'Baseline Reference: MSE, Grouping and Deviance '
                 f'[{scenario.name}, $\\kappa$={kappa:g}]',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.15)
    _scenario_annotation(
        ax, scenario, kappa,
        {'alpha': optimal_alpha, 'd': optimal_d,
         'lambda1': lambda1_opt, 'lambda2': lambda2_opt},
        'upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    saved: {save_path}")


#  CSV EXPORT & REPORT
def export_summary(summary_dict, filename, output_dir):
    if not summary_dict:
        return
    df = pd.DataFrame(summary_dict).T
    df.index.name = 'Method'
    path = output_dir / filename
    df.to_csv(path, float_format='%.4f')
    print(f"    exported: {path}")


def export_axis_stats(by_method, filename, output_dir, config,
                      proposed_key='Proposed',
                      extra_label_cols=None):
    """
    For EVERY numeric metric present in the per-replication records, compute
    and write to CSV:

    """

    ci_pct = int((1 - config.bootstrap_alpha) * 100)

    # ── collect all numeric metric names from the records ──────────────────
    all_metrics = []
    seen = set()
    for recs in by_method.values():
        for rec in recs:
            for k in rec:
                if k not in seen and not k.startswith('_'):
                    try:
                        v = rec[k]
                        float(v)          # keep only numeric scalars
                        seen.add(k)
                        all_metrics.append(k)
                    except (TypeError, ValueError):
                        seen.add(k)       # mark as seen but don't include
        if len(seen) > 50:               # safety cap
            break

    # preserve a stable order: compute_metrics order first, then extras
    _preferred = ['MSE', 'Deviance_per_obs', 'Group_Within', 'Group_External',
                  'Group_Noise', 'Deviance_total', 'MAE', 'RMSE',
                  'Bias_Squared', 'Variance', 'n_train', 'n_test']
    ordered = [m for m in _preferred if m in seen] + \
              [m for m in all_metrics if m not in _preferred]

    # ── reference values for Wilcoxon ──────────────────────────────────────
    ref_recs = by_method.get(proposed_key, [])
    proposed_vals = {
        metric: [r[metric] for r in ref_recs
                 if metric in r and _is_finite(r[metric])]
        for metric in ordered
    }

    rows = []
    for method, recs in by_method.items():
        if not recs:
            continue
        for metric in ordered:
            vals = [r[metric] for r in recs
                    if metric in r and _is_finite(r[metric])]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            mean_v, ci_lo, ci_hi = bootstrap_ci(
                arr, n_bootstrap=config.n_bootstrap,
                alpha=config.bootstrap_alpha)
            sd_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan

            wstat, wpval, wdir = np.nan, np.nan, ''
            if (method != proposed_key
                    and proposed_key != '__none__'
                    and proposed_vals.get(metric)):
                wstat, wpval, wdir = paired_wilcoxon(
                    proposed_vals[metric], vals)

            row = {
                'method':              method,
                'metric':              metric,
                'n_reps':              len(vals),
                'mean':                mean_v,
                'sd':                  sd_v,
                f'CI_{ci_pct}_lo':     ci_lo,
                f'CI_{ci_pct}_hi':     ci_hi,
                'wilcoxon_stat':       wstat,
                'wilcoxon_pval':       wpval,
                'wilcoxon_direction':  wdir,
            }
            if extra_label_cols:
                row.update(extra_label_cols)
            rows.append(row)

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not df.empty:
        df = add_holm_correction(df, family_cols=('metric',))
    # Write to disk only when a filename is given; callers that just want the
    # DataFrame (e.g. to concatenate per-n slices in memory) pass filename=None
    # to avoid temp-file round-trips.
    if filename is not None and not df.empty:
        path = output_dir / filename
        df.to_csv(path, index=False, float_format='%.6g')
        print(f"    exported: {path}")
    return df


def _is_finite(v):

    try:
        f = float(v)
        return np.isfinite(f)
    except (TypeError, ValueError):
        return False


def generate_findings(r1, r2, r3, r4, fixed_params, scenario_name, kappa):
    findings = []
    tag = f"[{scenario_name}, kappa={kappa}]"
    findings.append(f"{tag} Global parameters selected by repeated-CV mean deviance: "
                    f"alpha={fixed_params['alpha']:.3f}, d={fixed_params['d']:.3f}, "
                    f"lambda1={fixed_params['lambda1']:.4g}, lambda2={fixed_params['lambda2']:.4g}")
    if r1 and "by_d" in r1 and r1["by_d"]:
        d_vals = [d for d in r1["by_d"] if r1["by_d"][d]]
        if d_vals:
            dev_min = min(d_vals, key=lambda d: float(np.nanmean(
                [r["Deviance_per_obs"] for r in r1["by_d"][d]])))
            mse_min = min(d_vals, key=lambda d: float(np.nanmean(
                [r["MSE"] for r in r1["by_d"][d]])))
            grp_min = min(d_vals, key=lambda d: float(np.nanmean(
                [r["Group_Within"] for r in r1["by_d"][d]])))
            findings.append(f"{tag} d-sensitivity with alpha/lambda fixed: "
                            f"dev-min d={dev_min:.3f}, MSE-min d={mse_min:.3f}, grouping-min d={grp_min:.3f}")
    if r2 and "by_pivot" in r2:
        rg = r2["by_pivot"].get("ridge_pivot_reference", [])
        if rg:
            ref = float(np.nanmean([r["Deviance_per_obs"] for r in rg]))
            others = [float(np.nanmean([r["Deviance_per_obs"] for r in r2["by_pivot"][p]]))
                      for p in r2["by_pivot"]
                      if p != "ridge_pivot_reference" and r2["by_pivot"][p]]
            if others and ref > 1e-10:
                findings.append(f"{tag} Pivot robustness: worst-case deviance change = "
                                f"{(max(others) / ref - 1) * 100:+.1f}%")
    if r3 and "by_n" in r3:
        ns = sorted([n for n in r3["by_n"] if "Proposed" in r3["by_n"][n]
                     and r3["by_n"][n]["Proposed"]])
        if len(ns) >= 2:
            mse_p = [float(np.nanmean([r["MSE"] for r in r3["by_n"][n]["Proposed"]]))
                     for n in ns]
            if mse_p[-1] < mse_p[0] and mse_p[0] > 1e-10:
                findings.append(f"{tag} Proposed MSE shrinks {mse_p[0] / mse_p[-1]:.2f}x "
                                f"as n: {ns[0]}->{ns[-1]}")
    if r4 and "by_method" in r4:
        for metric, label, rule in [
            ('Deviance_per_obs', 'lowest deviance/n', min),
            ('MSE', 'lowest MSE', min),
            ('Group_Within', 'lowest within-group dispersion', min),
        ]:
            vals = {m: float(np.nanmean([r.get(metric, np.nan)
                                          for r in r4["by_method"][m]]))
                    for m in r4["by_method"] if r4["by_method"][m]}
            vals = {k: v for k, v in vals.items() if not np.isnan(v)}
            if vals:
                best = rule(vals, key=vals.get)
                findings.append(f"{tag} {label}: {best} ({vals[best]:.4f})")
    return findings


def generate_report(config, scenario, kappa, timing, findings, fixed_params, output_dir):
    mode = "FAST" if config.fast_mode else "FULL"
    regime = "convex-only (alpha in (1, 2])" if config.convex_only else \
             "full grid including concave (alpha < 1)"
    report = f"""
RUN REPORT v7-fixed - Convex Train/Test Pivot-Centered Shrinkage Estimator (Poisson)
Scenario: {scenario.label}
kappa (signal proportion): {kappa}
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURATION:
  Mode:           {mode}
  Numba:          {USE_NUMBA}
  Joblib:         {USE_JOBLIB}
  Parallel reps:  {config.parallel_reps}
  alpha regime:   {regime}
  alpha grid:     {config.alpha_grid}
  d grid:         {config.d_grid}
  Replications:   {config.n_replications}
  CV repeats:      {config.cv_repeats}
  Train n:        scenario n = {scenario.n_samples}
  Test size:      {config.test_size if config.test_size is not None else str(config.test_frac) + ' * train n'}
  CV folds:       {config.n_folds}
  Tolerance:      {config.tol}
  Seed base:      {config.seed_base}
  Axes run:       {', '.join(config.run_axes)}
  Baselines:      {'Lasso, ENet, MNet as reference only' if config.include_baselines else 'disabled'}
  Axis 5:         REMOVED (not present in this version)
  Axis 3b n vals: {config.axis3b_n_values}
  Bootstrap reps: {config.n_bootstrap}  (for CIs and significance)
  Bootstrap alpha:{config.bootstrap_alpha}  (→ {int((1-config.bootstrap_alpha)*100)}% CI)

SELECTED PARAMETERS BY REPEATED K-FOLD CV
(minimum mean deviance across repeat-level fold means):
  alpha*:          {fixed_params['alpha']:.6g}
  d*:              {fixed_params['d']:.6g}
  lambda1*:        {fixed_params['lambda1']:.6g}
  lambda2*:        {fixed_params['lambda2']:.6g}
  CV dev (mean):   {fixed_params.get('cv_deviance', float('nan')):.6g}
  CV dev SD:       {fixed_params.get('cv_deviance_sd', float('nan')):.6g}
  CV dev SE:       {fixed_params.get('cv_deviance_se', float('nan')):.6g}

TIMING (minutes):
  Selection: {timing.get('selection', 0)/60:.2f}
  Axis 1:    {timing.get('axis1', 0)/60:.2f}
  Axis 2:    {timing.get('axis2', 0)/60:.2f}
  Coef path: {timing.get('coef_path', 0)/60:.2f}
  Axis 3:    {timing.get('axis3', 0)/60:.2f}
  Axis 3b:   {timing.get('axis3b', 0)/60:.2f}
  Axis 4:    {timing.get('axis4', 0)/60:.2f}
  TOTAL :    {timing.get('total', 0)/60:.2f}

KEY FINDINGS:
"""
    for i, f in enumerate(findings, 1):
        report += f"  {i}. {f}\n"

    report += f"""



OUTPUT: {output_dir}/*.{{pdf,csv,txt}}
"""
    path = output_dir / 'run_report.txt'
    with open(path, 'w') as f:
        f.write(report)
    print(f"    report: {path}")


#  AGGREGATE BASELINE COMPARISON

def aggregate_baselines(r1, r3, r4):
    methods = ["Proposed", "Lasso", "ENet", "MNet"]
    out = {}
    if r4 and "by_method" in r4:
        for m in methods:
            if m in r4["by_method"] and r4["by_method"][m]:
                out[m] = r4["by_method"][m]
        if out:
            return out
    if r3 and "by_n" in r3:
        ns = sorted(r3["by_n"].keys(), reverse=True)
        for n in ns:
            cand = {m: r3["by_n"][n][m] for m in methods
                     if m in r3["by_n"][n] and r3["by_n"][n][m]}
            if len(cand) >= 4:
                return cand
    if r1:
        for m in methods:
            if m == "Proposed":
                if "mse_optimal_d" in r1 and r1["mse_optimal_d"] in r1.get("by_d", {}):
                    out[m] = r1["by_d"][r1["mse_optimal_d"]]
            elif m in r1.get("baselines", {}) and r1["baselines"][m]:
                out[m] = r1["baselines"][m]
    return out


#  MAIN
def run_one_setting(scenario_name, kappa, config):
    config.set_scenario(scenario_name)
    config.set_kappa(kappa)
    scenario = config.current_scenario

    setting_dir = config.output_dir / f"scenario_{scenario_name}_kappa{kappa}"
    setting_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"  Scenario {scenario_name} | kappa = {kappa} | "
          f"output: {setting_dir.name}")
    print(f"{'=' * 72}")

    timing = {}
    results = {}
    t_total = time.time()

    t0 = time.time()
    fixed_params, selection_table, selection_grouped = select_global_optimal_params(scenario, kappa, config)
    timing['selection'] = time.time() - t0
    results['selection'] = {
        'params': fixed_params,
        'table': selection_table,
        'grouped': selection_grouped,
    }
    selection_table.to_csv(setting_dir / 'repeated_cv_fold_results.csv', index=False, float_format='%.8g')
    selection_grouped.to_csv(setting_dir / 'repeated_cv_candidate_summary.csv', index=False, float_format='%.8g')

    if 'axis1' in config.run_axes:
        t0 = time.time()
        results['axis1'] = run_axis1(scenario, kappa, fixed_params, config)
        timing['axis1'] = time.time() - t0
        plot_axis1(results['axis1']['by_d'], scenario, kappa,
                    fixed_params['alpha'], fixed_params['d'],
                    fixed_params['lambda1'], fixed_params['lambda2'],
                    setting_dir / 'fig_axis1_sensitivity.pdf')
        export_summary(results['axis1']['summary'], 'axis1_summary.csv', setting_dir)
        export_axis_stats(results['axis1']['_by_label'], 'axis1_stats.csv',
                          setting_dir, config,
                          proposed_key=f"Proposed(d={fixed_params['d']:.3f})")

    if 'axis2' in config.run_axes:
        t0 = time.time()
        results['axis2'] = run_axis2(scenario, kappa, fixed_params, config)
        timing['axis2'] = time.time() - t0
        plot_axis2(results['axis2']['by_pivot'], scenario, kappa,
                    fixed_params['alpha'], fixed_params['d'],
                    fixed_params['lambda1'], fixed_params['lambda2'],
                    setting_dir / 'fig_axis2_robustness.pdf')
        export_summary(results['axis2']['summary'], 'axis2_summary.csv', setting_dir)
        export_axis_stats(results['axis2']['_by_label'], 'axis2_stats.csv',
                          setting_dir, config,
                          proposed_key='ridge_pivot_reference')

    if 'coef_path' in config.run_axes:
        t0 = time.time()
        results['coef_path'] = run_coefficient_path(
            scenario, kappa, fixed_params, config, setting_dir)
        timing['coef_path'] = time.time() - t0

    if 'axis3' in config.run_axes:
        t0 = time.time()
        results['axis3'] = run_axis3(scenario, kappa, fixed_params, config)
        timing['axis3'] = time.time() - t0
        plot_axis3(results['axis3']['by_n'], scenario, kappa,
                    results['axis3']['optimal_params'],
                    setting_dir / 'fig_axis3a_classical_sample_size.pdf')
        export_summary(results['axis3']['summary'], 'axis3_summary.csv', setting_dir)


        _a3_frames = []
        for n_obs in (config.axis3_n_values or []):
            by_n_slice = {
                m: results['axis3']['by_n'][n_obs][m]
                for m in results['axis3']['by_n'].get(n_obs, {})
                if results['axis3']['by_n'][n_obs][m]
            }
            if not by_n_slice:
                continue
            _df_slice = export_axis_stats(
                by_n_slice, None, setting_dir, config,
                proposed_key='Proposed',
                extra_label_cols={'n': n_obs,
                                  'p': scenario.p_total,
                                  'p_over_n': round(scenario.p_total / max(n_obs, 1), 4)})
            if not _df_slice.empty:
                _a3_frames.append(_df_slice)
        if _a3_frames:
            _df_a3 = pd.concat(_a3_frames, ignore_index=True)
            _a3_path = setting_dir / 'axis3_stats.csv'
            _df_a3.to_csv(_a3_path, index=False, float_format='%.6g')
            print(f"    exported: {_a3_path}")

    if 'axis3b' in config.run_axes:
        t0 = time.time()
        results['axis3b'] = run_axis3b(scenario, kappa, fixed_params, config, setting_dir)
        timing['axis3b'] = time.time() - t0

    if 'axis4' in config.run_axes:
        t0 = time.time()
        results['axis4'] = run_axis4(scenario, kappa, fixed_params, config)
        timing['axis4'] = time.time() - t0
        plot_axis4(results['axis4']['by_method'], results['axis4']['masks'],
                    results['axis4']['bias_var'], scenario, kappa,
                    fixed_params['alpha'], fixed_params['d'],
                    fixed_params['lambda1'], fixed_params['lambda2'],
                    setting_dir / 'fig_axis4_grouping.pdf')
        bv = results['axis4']['bias_var']
        if bv:
            pd.DataFrame([
                {'Method': m, 'Bias_Squared': v['Bias_Squared'],
                 'Variance': v['Variance'], 'MSE': v['MSE']}
                for m, v in bv.items()
            ]).to_csv(setting_dir / 'axis4_bias_variance.csv',
                      index=False, float_format='%.6g')
            print(f"    exported: {setting_dir / 'axis4_bias_variance.csv'}")
        export_summary(results['axis4']['summary'], 'axis4_summary.csv', setting_dir)
        # significance testing for axis4
        export_axis4_significance(results['axis4']['by_method'], config, setting_dir)

    if 'baseline' in config.run_axes and config.include_baselines:
        baselines = aggregate_baselines(
            results.get('axis1'), results.get('axis3'), results.get('axis4'))
        if baselines:
            plot_baseline_comparison(
                baselines, scenario, kappa,
                fixed_params['alpha'], fixed_params['d'],
                fixed_params['lambda1'], fixed_params['lambda2'],
                setting_dir / 'fig_baseline_comparison.pdf')
            summary = {m: {k: float(np.nanmean([r[k] for r in baselines[m]
                                                  if not k.startswith('_')]))
                            for k in baselines[m][0] if not k.startswith('_')}
                       for m in baselines if baselines[m]}
            export_summary(summary, 'baseline_comparison.csv', setting_dir)

    #  POST-RUN AXIS 3A / AXIS 3B

    if config.post_run_axes:
        print(
            "\n  [Post-run] Main-axis outputs saved. "
            "Starting Axis 3a and Axis 3b."
        )

    if 'axis3' in config.post_run_axes and 'axis3' not in results:
        t0 = time.time()
        results['axis3'] = run_axis3(scenario, kappa, fixed_params, config)
        timing['axis3'] = time.time() - t0
        plot_axis3(
            results['axis3']['by_n'],
            scenario,
            kappa,
            results['axis3']['optimal_params'],
            setting_dir / 'fig_axis3a_classical_sample_size.pdf',
        )
        export_summary(
            results['axis3']['summary'],
            'axis3_summary.csv',
            setting_dir,
        )

        _a3_frames = []
        for n_obs in (config.axis3_n_values or []):
            by_n_slice = {
                m: results['axis3']['by_n'][n_obs][m]
                for m in results['axis3']['by_n'].get(n_obs, {})
                if results['axis3']['by_n'][n_obs][m]
            }
            if not by_n_slice:
                continue

            _df_slice = export_axis_stats(
                by_n_slice,
                None,
                setting_dir,
                config,
                proposed_key='Proposed',
                extra_label_cols={
                    'n': n_obs,
                    'p': scenario.p_total,
                    'p_over_n': round(
                        scenario.p_total / max(n_obs, 1), 4
                    ),
                },
            )
            if not _df_slice.empty:
                _a3_frames.append(_df_slice)

        if _a3_frames:
            _df_a3 = pd.concat(_a3_frames, ignore_index=True)
            _a3_path = setting_dir / 'axis3_stats.csv'
            _df_a3.to_csv(
                _a3_path,
                index=False,
                float_format='%.6g',
            )
            print(f"    exported: {_a3_path}")

    if 'axis3b' in config.post_run_axes and 'axis3b' not in results:
        t0 = time.time()
        results['axis3b'] = run_axis3b(
            scenario,
            kappa,
            fixed_params,
            config,
            setting_dir,
        )
        timing['axis3b'] = time.time() - t0

    timing['total'] = time.time() - t_total
    findings = generate_findings(results.get('axis1'), results.get('axis2'),
                                   results.get('axis3'), results.get('axis4'),
                                   fixed_params, scenario_name, kappa)
    generate_report(config, scenario, kappa, timing, findings, fixed_params, setting_dir)
    print(f"\n  Setting done in {timing['total'] / 60:.1f} min")
    print(f"  Output: {setting_dir.absolute()}")
    return {'scenario': scenario_name, 'kappa': kappa,
            'results': results, 'timing': timing}


def main():
    global CFG
    if CFG is None:
        CFG = ExperimentConfig()

    print(f"\n{'=' * 72}")
    print(f"  PIVOT ESTIMATOR v7-fixed  Convex Train/Test + CV Validation")
    print(f"{'=' * 72}")
    print(f"  fast_mode      : {CFG.fast_mode}")
    print(f"  convex_only    : {CFG.convex_only}")
    print(f"  alpha grid     : {CFG.alpha_grid}")
    print(f"  d grid         : {CFG.d_grid}")
    print(f"  kappa grid     : {CFG.kappa_grid}")
    print(f"  scenarios      : {CFG.scenarios_to_run}")
    print(f"  axes           : {CFG.run_axes}")
    print(f"  axis 3b n vals : {CFG.axis3b_n_values}")
    print(f"  CV repeats     : {CFG.cv_repeats}")
    print(f"  baselines      : {CFG.include_baselines}")
    print(f"  replications   : {CFG.n_replications}")
    print(f"  train/test     : train n = scenario n; test = {CFG.test_size if CFG.test_size is not None else str(CFG.test_frac) + ' * train n'}")
    print(f"  CV folds       : {CFG.n_folds}")
    print(f"  bootstrap reps : {CFG.n_bootstrap}")
    print(f"  numba          : {USE_NUMBA}")
    print(f"  joblib         : {USE_JOBLIB}")
    print(f"  parallel reps  : {CFG.parallel_reps}")
    print(f"  worker limit   : {CFG.max_parallel_jobs if CFG.max_parallel_jobs is not None else 'auto (CPU count - 1)'}")
    print(f"  backend        : {CFG.parallel_backend}")
    print(f"  BLAS threads/worker: {CFG.inner_max_num_threads}")
    print(f"  output_dir     : {CFG.output_dir.absolute()}")
    print(f"{'=' * 72}\n")

    all_results = []
    t_global = time.time()
    for scen_name in CFG.scenarios_to_run:
        for kappa in CFG.kappa_grid:
            r = run_one_setting(scen_name, kappa, CFG)
            all_results.append(r)

    print(f"\n{'=' * 72}")
    print(f"  GLOBAL SUMMARY  (total time: {(time.time() - t_global) / 60:.1f} min)")
    print(f"{'=' * 72}")
    rows = []
    for r in all_results:
        params = r['results'].get('selection', {}).get('params', {})
        rows.append({
            'scenario': r['scenario'],
            'kappa': r['kappa'],
            'alpha_opt': params.get('alpha', float('nan')),
            'd_opt_cv_deviance': params.get('d', float('nan')),
            'lambda1_opt': params.get('lambda1', float('nan')),
            'lambda2_opt': params.get('lambda2', float('nan')),
            'cv_deviance_mean': params.get('cv_deviance', float('nan')),
            'cv_deviance_sd': params.get('cv_deviance_sd', float('nan')),
            'cv_deviance_se': params.get('cv_deviance_se', float('nan')),
            'time_min': r['timing'].get('total', 0) / 60,
        })
    df = pd.DataFrame(rows)
    summary_path = CFG.output_dir / 'global_summary.csv'
    df.to_csv(summary_path, index=False, float_format='%.4f')
    print(f"\n{df.to_string(index=False)}")
    print(f"\n  Cross-setting summary written: {summary_path}")
    print(f"\n  All outputs: {CFG.output_dir.absolute()}\n")


if __name__ == "__main__":
    main()

