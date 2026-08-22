from pathlib import Path

import numpy as np
import pandas as pd

from decision_intelligence.experiment import calibration_ece, choose_threshold, utility_if_action


def test_utility_prefers_higher_probability() -> None:
    low = utility_if_action(np.array([0.10]), 100.0, 10.0, 30.0)[0]
    high = utility_if_action(np.array([0.90]), 100.0, 10.0, 30.0)[0]
    assert high > low


def test_ece_zero_for_perfectly_calibrated_single_bin() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert calibration_ece(y, p, bins=2) == 0.0


def test_threshold_selection_returns_configured_operating_point() -> None:
    cfg = {
        "decision_grid_min": 0.05,
        "decision_grid_max": 0.95,
        "decision_grid_step": 0.01,
    }
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.05, 0.20, 0.70, 0.80, 0.90])
    threshold, table = choose_threshold(y, p, cfg, 100.0, 10.0, 30.0)
    assert 0.05 <= threshold <= 0.95
    assert table["is_primary"].sum() == 1


def test_project_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "config" / "experiment.json").exists()
    assert (root / "paper" / "research_framework.md").exists()
