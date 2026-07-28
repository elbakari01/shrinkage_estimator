from pathlib import Path

import numpy as np

from shrinkage_estimator.simulation import (
    ExperimentConfig,
    SCENARIOS,
    generate_covariance,
    stable_seed,
)


def test_stable_seed_is_reproducible():
    first = stable_seed(1992, "scenario", "D", 0.08, 1)
    second = stable_seed(1992, "scenario", "D", 0.08, 1)
    assert first == second
    assert 0 <= first < 2**32 - 1


def test_scenario_dimensions_are_consistent():
    for scenario in SCENARIOS.values():
        assert scenario.p_total == sum(scenario.p_groups)
        assert scenario.n_samples > 0


def test_generated_covariance_is_symmetric_positive_semidefinite():
    scenario = SCENARIOS["D"]
    sigma, starts = generate_covariance(
        scenario.p_groups,
        scenario.rho_within,
        scenario.rho_between,
        scenario.rho_noise,
    )
    assert sigma.shape == (scenario.p_total, scenario.p_total)
    assert np.allclose(sigma, sigma.T)
    assert np.linalg.eigvalsh(sigma).min() >= -1e-8
    assert starts[-1] == scenario.p_total


def test_fast_configuration_uses_small_diagnostic_settings(tmp_path: Path):
    config = ExperimentConfig(fast_mode=True, output_dir=tmp_path / "results")
    assert config.n_replications == 4
    assert config.n_folds == 3
    assert config.cv_repeats == 2
