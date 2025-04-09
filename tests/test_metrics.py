from pathlib import Path

import numpy as np
import pandas as pd
import pyrallis
import pytest
from generation.metrics.metrics import (
    Accuracy,
    Levenshtein,
    F1Metric,
    Gini,
    Reconstruction,
    ShannonEntropy,
)
from main import PipelineConfig


def create_multiple_data(
    y_true_list: list[list[int]], y_pred_list: list[list[int]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert len(y_true_list) == len(
        y_pred_list
    ), "y_true and y_pred must have the same length"

    gt = pd.DataFrame(
        [
            {"client_id": i, "event_type": np.array(y_true_list[i])}
            for i in range(len(y_true_list))
        ]
    )

    gen = pd.DataFrame(
        [
            {"client_id": i, "event_type": np.array(y_pred_list[i])}
            for i in range(len(y_pred_list))
        ]
    )

    return gt, gen


def create_multiple_features_data(
    y_true_list: list[dict], y_pred_list: list[dict]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert len(y_true_list) == len(
        y_pred_list
    ), "y_true and y_pred must have the same length"
    gt = pd.DataFrame(
        [
            {"client_id": i, **el}
            for i, el in enumerate(y_true_list)
        ]
    )

    gen = pd.DataFrame(
        [
            {"client_id": i, **el}
            for i, el in enumerate(y_pred_list)
        ]
    )

    return gt, gen


@pytest.fixture
def config() -> PipelineConfig:
    return pyrallis.parse(
        args=["--config", "spec_config.yaml"], config_class=PipelineConfig
    )

@pytest.mark.parametrize(
    "y_true, y_pred, expected_est",
    [
[
        [
            {"event_type": [0, 0, 0, 0], "src_type32": [1, 2, 3, 1], "amount": [1122, 1313, 13121, 100]},
            {"event_type": [0, 1, 1, 0], "src_type32": [1, 2, 3, 1], "amount": [1122, 1313, 13121, 100]},
        ],
        [
            {"event_type": [0, 0, 0, 0], "src_type32": [1, 2, 3, 1], "amount": [1122, 1313, 13121, 100]},
            {"event_type": [0, 1, 1, 0], "src_type32": [1, 2, 3, 1], "amount": [1122, 1313, 13121, 100]},
        ],
        (1 + 1 + 0) / 3
    ],
    [
        [
            {"event_type": [0, 0, 0, 0], "src_type32": [0, 0, 0, 0], "amount": [1122, 1313, 13121, 100]},
            {"event_type": [0, 1, 1, 0], "src_type32": [1, 2, 3, 1], "amount": [1122, 1313, 13121, 100]},
        ],
        [
            {"event_type": [0, 0, 0, 0], "src_type32": [1, 2, 3, 1], "amount": [1122, 1313, 13121, 100]},
            {"event_type": [0, 1, 1, 0], "src_type32": [1, 2, 3, 1], "amount": [1122, 1313, 13121, 100]},
        ],
        (1 + 0.5 + 0) / 3
    ],
    [
        [
            {"event_type": [0, 0, 0, 0], "src_type32": [0, 0, 0, 0], "amount": [0, 1, 2, 3]},
            {"event_type": [0, 1, 1, 0], "src_type32": [1, 2, 3, 1], "amount": [1, 3, 5, 1]},
        ],
        [
            {"event_type": [0, 0, 0, 0], "src_type32": [1, 2, 3, 1], "amount": [1, 1, 2, 3]},
            {"event_type": [0, 1, 0, 0], "src_type32": [1, 2, 3, 1], "amount": [1, 3, 5, 1]},
        ],
        ((1 + 0.75) / 2 + 0.5 + (0.25 + 0) / 2) / 3
    ],
    ],
    
    ids=[
        "perfect",
        "src_type_fail",
        "all_features_fail"
    ],
)
def test_reconstruction_metric(
    y_true, y_pred, expected_est, config: PipelineConfig
):
    log_dir = Path("tests/log")
    log_dir.mkdir(parents=True, exist_ok=True)
    gt, gen = create_multiple_features_data(y_true, y_pred)

    metric = Reconstruction(
        devices=["cpu"],
        data_conf=config.data_conf,
        log_dir=Path("."),
    )

    est = metric(gt, gen)

    assert np.isclose(
        np.mean(list(est.values())), expected_est, atol=1e-5
    ), f"Expected {expected_est}, got {est}"


@pytest.mark.parametrize(
    "y_true, y_pred, expected_est, expected_overall_est",
    [
        ([[1, 2]], [[1, 2]], 0.0, 0.0),
        (
            [[1, 2]],
            [[1, 1]],
            ((-1 * np.log2(1)) - (-0.5 * np.log2(0.5) - 0.5 * np.log2(0.5)))
            / abs(-0.5 * np.log2(0.5) - 0.5 * np.log2(0.5)),
            ((-1 * np.log2(1)) - (-0.5 * np.log2(0.5) - 0.5 * np.log2(0.5)))
            / abs(-0.5 * np.log2(0.5) - 0.5 * np.log2(0.5)),
        ),
        (
            [[0, 1, 1, 1]],
            [[0, 0, 1, 1]],
            (
                -(0.5 * np.log2(0.5) + 0.5 * np.log2(0.5))
                + (0.25 * np.log2(0.25) + 0.75 * np.log2(0.75))
            )
            / abs((0.25 * np.log2(0.25) + 0.75 * np.log2(0.75))),
            (
                (-0.5 * np.log2(0.5) - 0.5 * np.log2(0.5))
                - (-0.25 * np.log2(0.25) - 0.75 * np.log2(0.75))
            )
            / abs((-0.25 * np.log2(0.25) - 0.75 * np.log2(0.75))),
        ),
    ],
    ids=[
        "perfect",
        "only_one_pred",
        "imbalanced",
    ],
)
def test_shannon_entropy_metric(
    y_true, y_pred, expected_est, expected_overall_est, config: PipelineConfig
):
    log_dir = Path("tests/log")
    log_dir.mkdir(parents=True, exist_ok=True)
    gt, gen = create_multiple_data(y_true, y_pred)

    metric = ShannonEntropy(
        devices=["cpu"],
        data_conf=config.data_conf,
        log_dir=Path("."),
        target_key="event_type",
        overall=False,
    )

    est = metric(gt, gen)

    assert np.isclose(
        est["relative"], expected_est, atol=1e-5
    ), f"Expected {expected_est}, got {est}"

    metric = ShannonEntropy(
        devices=["cuda:0"],
        data_conf=config.data_conf,
        log_dir=log_dir,
        target_key="event_type",
        overall=True,
    )

    est = metric(gt, gen)

    assert np.isclose(
        est["relative"], expected_overall_est, atol=1e-5
    ), f"Expected {expected_overall_est}, got {est}"


@pytest.mark.parametrize(
    "y_true, y_pred, expected_est, expected_overall_est",
    [
        ([[1, 2]], [[1, 2]], 0.0, 0.0),
        ([[1, 2]], [[1, 1]], 1.0, 1.0),
        ([[0, 1, 1, 1]], [[0, 0, 1, 1]], -1.0, -1.0),
        (
            [
                [0, 1, 2, 3, 4, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 1],
                [1, 2, 2, 3, 2, 1, 2, 0, 1],
            ],
            [
                [0, 1, 2, 3, 4, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 1],
                [1, 2, 2, 3, 2, 2, 2, 2, 1],
            ],
            0.83333,
            -3 / 5,
        ),
    ],
    ids=["perfect", "only_one_pred", "imbalanced", "tree clients"],
)
def test_gini_metric(
    y_true, y_pred, expected_est, expected_overall_est, config: PipelineConfig
):
    log_dir = Path("tests/log")
    log_dir.mkdir(parents=True, exist_ok=True)
    gt, gen = create_multiple_data(y_true, y_pred)

    gini = Gini(
        devices=["cuda:0"],
        data_conf=config.data_conf,
        log_dir=log_dir,
        target_key="event_type",
        overall=False,
    )
    est = gini(gt, gen)
    print(est)
    assert np.isclose(
        est["relative"], expected_est, atol=1e-5
    ), f"Expected {expected_est}, got {est}"

    gini = Gini(
        devices=["cuda:0"],
        data_conf=config.data_conf,
        log_dir=log_dir,
        target_key="event_type",
        overall=True,
    )

    est = gini(gt, gen)

    assert np.isclose(
        est["relative"], expected_overall_est, atol=1e-5
    ), f"Expected {expected_est}, got {est}"


# mini-test for f1-metric at first
def test_get_statistics_unordered():
    metric = F1Metric(
        average="macro",
        devices=["cpu"],
        data_conf=None,
        log_dir=Path("."),
        target_key="event_type",
    )

    gt = np.array([1, 1, 2, 3])
    pred = np.array([1, 2, 2, 4])

    stats = metric.get_statistics(gt, pred)

    s1 = stats[1]
    assert s1["tp"] == 1
    assert s1["fp"] == 0
    assert s1["fn"] == 1
    assert np.isclose(s1["precision"], 1.0)
    assert np.isclose(s1["recall"], 0.5)


@pytest.mark.parametrize(
    "y_true, y_pred, expected_macro, expected_micro",
    [
        ([[1, 2, 3, 4]], [[1, 2, 3, 4]], 1.0, 1.0),  # perfect
        ([[1, 1, 2, 3]], [[1, 1, 2, 4]], 0.5, 0.75),  # partial_class
        ([[1, 2, 3, 4]], [[1, 1, 1, 1]], 0.1, 0.25),  # imbalanced
        ([[1, 2, 3]], [[]], 0.0, 0.0),  # empty_pred
        ([[]], [[]], 1.0, 1.0),  # empty_both
        (
            [
                [0, 1, 2, 3, 4, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 1],
                [1, 2, 2, 3, 2, 1, 2, 0, 1],
            ],
            [
                [0, 1, 2, 3, 4, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 1],
                [1, 2, 2, 3, 2, 2, 2, 2, 1],
            ],
            0.707407407,
            0.75,
        ),
    ],
    ids=[
        "perfect",
        "partial_class",
        "imbalanced",
        "empty_pred",
        "empty_both",
        "three users",
    ],
)
def test_f1_metric(
    y_true, y_pred, expected_macro, expected_micro, config: PipelineConfig
):
    gt, gen = create_multiple_data(y_true, y_pred)

    f1_macro = F1Metric(
        average="macro",
        devices=["cpu"],
        data_conf=config.data_conf,
        log_dir=Path("."),
        target_key="event_type",
    )
    score_macro = f1_macro(gt, gen)
    assert np.isclose(
        score_macro, expected_macro, atol=1e-5
    ), f"Macro F1: expected {expected_macro}, got {score_macro}"

    f1_micro = F1Metric(
        average="micro",
        devices=["cpu"],
        data_conf=config.data_conf,
        log_dir=Path("."),
        target_key="event_type",
    )
    score_micro = f1_micro(gt, gen)
    assert np.isclose(
        score_micro, expected_micro, atol=1e-5
    ), f"Micro F1: expected {expected_micro}, got {score_micro}"


@pytest.mark.parametrize(
    "y_true, y_pred, expected_acc",
    [
        ([[1, 2, 3, 0, 1, 2, 3]], [[1, 2, 3, 0, 1, 2, 3]], 1.0),
        ([[1, 2, 3, 0, 1, 2, 3]], [[1, 2, 3, 0, 1, 2, 4]], 0.75),
        ([[1, 2, 3, 0, 0, 0, 0]], [[1, 2, 3, 1, 1, 1, 1]], 0.0),
        (
            [[0, 0, 0, 0, 0, 0, 0]],
            [[9, 9, 9, 0, 0, 0, 0]],
            1.0,
        ),  # Ensure that only the last (gen_len=4) values affect the metric.
        (
            [
                [0, 1, 2, 3, 4, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 1],
                [1, 2, 2, 3, 2, 1, 2, 0, 1],
            ],
            [
                [0, 1, 2, 3, 4, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 1],
                [1, 2, 2, 3, 2, 2, 2, 2, 1],
            ],
            ((3 / 4 + 1 + 2 / 4) / 3.0),
        ),
    ],
)
def test_accuracy_metric(y_true, y_pred, expected_acc, config: PipelineConfig):
    log_dir = Path("tests/log")
    log_dir.mkdir(parents=True, exist_ok=True)
    gt, gen = create_multiple_data(y_true, y_pred)

    accuracy = Accuracy(
        devices=['cuda:0'],
        data_conf=config.data_conf,
        log_dir=log_dir,
        target_key="event_type",
    )
    acc = accuracy(gt, gen)

    assert np.isclose(
        acc, expected_acc, atol=1e-5
    ), f"Expected {expected_acc}, got {acc}"


@pytest.mark.parametrize(
    "y_true, y_pred, expected_dist",
    [
        ([[1, 2, 3, 4]], [[1, 2, 3, 4]], 1 - 0 / 4),  # perfect match
        ([[1, 2, 3, 4]], [[1, 2, 3, 5]], 1 - 1 / 4),  # substitution
        ([[1, 2, 3, 4]], [[1, 2, 3]], 1 - 1 / 4),  # deletion
        ([[1, 2, 3]], [[1, 2, 3, 4]], 1 - 1 / 4),  # insertion
        ([[1, 1, 1, 1]], [[2, 2, 2, 2]], 1 - 4 / 4),  # total mismatch
        (
            [
                [0, 1, 2, 3, 4, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 1],
                [1, 2, 2, 3, 2, 1, 2, 0, 1],
            ],
            [
                [0, 1, 2, 3, 4, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 1],
                [1, 2, 2, 3, 2, 2, 2, 2, 1],
            ],
            0.75,
        ),
    ],
    ids=[
        "exact match",
        "1 substitution",
        "1 deletion",
        "1 insertion",
        "completely different",
        "tree clients",
    ],
)
def test_levenshtein_metric(y_true, y_pred, expected_dist, config: PipelineConfig):
    log_dir = Path("tests/log")
    log_dir.mkdir(parents=True, exist_ok=True)
    gt, gen = create_multiple_data(y_true, y_pred)

    lev = Levenshtein(
        devices=['cuda:0'],
        data_conf=config.data_conf,
        log_dir=log_dir,
        target_key="event_type",
    )

    dist = lev(gt, gen)

    assert np.isclose(
        dist, expected_dist, atol=1e-5
    ), f"Expected {expected_dist}, got {dist}"
