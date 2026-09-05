# %% IMPORTS

import typing as T

import numpy as np
import pandas as pd
import pydantic as pdt
import pytest

from bikes.core import schemas
from bikes.utils import splitters

# %% SPLITTERS


def test_train_test_splitter(inputs: schemas.Inputs, targets: schemas.Targets) -> None:
    # given
    test_size = 50
    random_state = 0
    splitter = splitters.TrainTestSplitter(
        shuffle=False, test_size=test_size, random_state=random_state
    )
    # when
    n_splits = splitter.get_n_splits(inputs=inputs, targets=targets)
    splits = list(splitter.split(inputs=inputs, targets=targets))
    train_index, test_index = splits[0]  # train/test indexes
    # then
    assert n_splits == len(splits) == 1, "Splitter should return 1 split!"
    assert len(test_index) == test_size, "Test index should have the given size!"
    assert len(train_index) == len(targets) - test_size, (
        "Train index should have the remaining size!"
    )
    assert not inputs.iloc[test_index].empty, "Test index should be a subset of the inputs!"
    assert not targets.iloc[train_index].empty, "Train index should be a subset of the targets!"


def test_time_series_splitter(inputs: schemas.Inputs, targets: schemas.Targets) -> None:
    # given
    gap = 0
    n_splits = 3
    test_size = 50
    splitter = splitters.TimeSeriesSplitter(gap=gap, n_splits=n_splits, test_size=test_size)
    # when
    n_splits = splitter.get_n_splits(inputs=inputs, targets=targets)
    splits = list(splitter.split(inputs=inputs, targets=targets))
    # then
    assert n_splits == len(splits), "Splitter should return the given n splits!"
    for i, (train_index, test_index) in enumerate(splits):
        assert len(test_index) == test_size, "Test index should have the given test size!"
        assert len(train_index) == (len(inputs) - test_size * (n_splits - i)), (
            "Train index should have the cumulative remaining size!"
        )
        assert train_index.max() < test_index.min(), (
            "Train index should always be lower than test index!"
        )
        assert not inputs.iloc[train_index].empty, "Train index should be a subset of the inputs!"
        assert not inputs.iloc[test_index].empty, "Test index should be a subset of the inputs!"


@pytest.mark.parametrize(
    "splitter",
    [splitters.TrainTestSplitter(test_size=10), splitters.TimeSeriesSplitter(test_size=10)],
)
@pytest.mark.parametrize(
    "invalid",
    [
        "reordered_targets",
        "missing_target",
        "duplicates",
        "empty",
        "unsorted",
        "same_hour",
        "non_midnight",
    ],
)
def test_splitters_reject_invalid_rows(
    inputs: schemas.Inputs,
    targets: schemas.Targets,
    splitter: splitters.Splitter,
    invalid: str,
) -> None:
    x, y = pd.DataFrame(inputs.copy()), pd.DataFrame(targets.copy())
    if invalid == "reordered_targets":
        y = y.iloc[::-1]
    elif invalid == "missing_target":
        y = y.iloc[:-1]
    elif invalid == "duplicates":
        x.index = y.index = pd.Index([0] * len(x))
    elif invalid == "empty":
        x, y = x.iloc[:0], y.iloc[:0]
    elif invalid == "unsorted":
        # Keep row pairing intact so only the chronology check can reject this.
        x, y = x.iloc[::-1], y.iloc[::-1]
    elif invalid == "non_midnight":
        x["dteday"] = x["dteday"] + pd.Timedelta(minutes=30)
    else:
        x.loc[x.index[1], "dteday"] = x.loc[x.index[0], "dteday"]
        x.loc[x.index[1], "hr"] = x.loc[x.index[0], "hr"]
    with pytest.raises(ValueError):
        list(splitter.split(T.cast(schemas.Inputs, x), T.cast(schemas.Targets, y)))


def test_temporal_folds_use_calendar_time_and_observation_gaps(
    inputs: schemas.Inputs, targets: schemas.Targets
) -> None:
    # Irregular sampling and decreasing source IDs must not alter calendar ordering.
    x = T.cast(schemas.Inputs, inputs.iloc[::2].copy())
    y = T.cast(schemas.Targets, targets.iloc[::2].copy())
    x.index = y.index = pd.Index(np.arange(len(x), 0, -1), dtype="uint32")
    timestamps = x["dteday"] + pd.to_timedelta(x["hr"].astype("int64"), unit="h")
    splitter = splitters.TimeSeriesSplitter(n_splits=3, test_size=10, gap=3)
    for train, test in splitter.split(x, y):
        assert not np.intersect1d(train, test).size
        assert train[-1] + 4 == test[0]
        assert timestamps.iloc[train].max() < timestamps.iloc[test].min()


def test_shuffle_is_rejected() -> None:
    with pytest.raises(pdt.ValidationError):
        splitters.TrainTestSplitter.model_validate({"shuffle": True})


@pytest.mark.parametrize(
    "config", [{"gap": -1}, {"n_splits": 1}, {"test_size": 0}, {"test_size": 0.5}]
)
def test_invalid_time_series_configuration(config: dict[str, int | float]) -> None:
    with pytest.raises(pdt.ValidationError):
        splitters.TimeSeriesSplitter.model_validate(config)
