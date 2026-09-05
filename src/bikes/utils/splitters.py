"""Split dataframes into subsets (e.g., train/valid/test)."""

# %% IMPORTS

from __future__ import annotations

import abc
import typing as T

import numpy as np
import numpy.typing as npt
import pandas as pd
import pydantic as pdt
from sklearn import model_selection

from bikes.core import schemas

# %% TYPES

Index = npt.NDArray[np.int64]
TrainTestIndex = tuple[Index, Index]
TrainTestSplits = T.Iterator[TrainTestIndex]


def check_chronological_inputs(inputs: schemas.Inputs, targets: schemas.Targets) -> None:
    """Reject unaligned rows and require strictly increasing calendar hours.

    Missing hours are allowed. Split sizes and gaps count observations, not elapsed
    hours. Source IDs identify rows; they do not establish chronological order.
    """
    schemas.check_row_alignment(inputs, targets)
    _calendar_hours(inputs)


def _calendar_hours(inputs: schemas.Inputs) -> pd.Series[pd.Timestamp]:
    """Validate a dataset and return its strictly ordered calendar hours."""
    schemas.check_row_alignment(inputs, inputs)
    checked = schemas.InputsSchema.check(inputs)
    dates = checked["dteday"]
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError("dteday must contain dates at midnight; hr supplies the hour.")
    timestamps = dates + pd.to_timedelta(checked["hr"].astype("int64"), unit="h")
    if not timestamps.is_unique or not timestamps.is_monotonic_increasing:
        raise ValueError("Calendar hours must be unique and strictly increasing.")

    return timestamps


def check_temporal_boundary(reference: schemas.Inputs, evaluation: schemas.Inputs) -> None:
    """Require evaluation IDs to be disjoint and all hours later than the reference.

    This checks supplied datasets, not a registered model’s training provenance.
    """
    reference_hours = _calendar_hours(reference)
    evaluation_hours = _calendar_hours(evaluation)
    if not reference.index.intersection(evaluation.index).empty:
        raise ValueError("Reference and evaluation row IDs must be disjoint.")
    if reference_hours.iloc[-1] >= evaluation_hours.iloc[0]:
        raise ValueError("Evaluation hours must be strictly later than all reference hours.")


# %% SPLITTERS


class Splitter(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a splitter.

    Use splitters to split data in sets.
    e.g., split between a train/test subsets.

    # https://scikit-learn.org/stable/glossary.html#term-CV-splitter
    """

    KIND: str

    @abc.abstractmethod
    def split(
        self,
        inputs: schemas.Inputs,
        targets: schemas.Targets,
        groups: Index | None = None,
    ) -> TrainTestSplits:
        """Split a dataframe into subsets.

        Args:
            inputs (schemas.Inputs): model inputs.
            targets (schemas.Targets): model targets.
            groups (Index | None, optional): group labels.

        Returns:
            TrainTestSplits: iterator over the dataframe train/test splits.
        """

    @abc.abstractmethod
    def get_n_splits(
        self,
        inputs: schemas.Inputs,
        targets: schemas.Targets,
        groups: Index | None = None,
    ) -> int:
        """Get the number of splits generated.

        Args:
            inputs (schemas.Inputs): models inputs.
            targets (schemas.Targets): model targets.
            groups (Index | None, optional): group labels.

        Returns:
            int: number of splits generated.
        """


class TrainTestSplitter(Splitter):
    """Split a dataframe into a train and test set.

    Parameters:
        shuffle (bool): must be False for the chronological bike example.
        test_size (int | float): number/ratio for the test set.
        random_state (int): random state for the splitter object.
    """

    KIND: T.Literal["TrainTestSplitter"] = "TrainTestSplitter"

    shuffle: T.Literal[False] = False
    test_size: int | float = 24 * 30 * 2  # 1440 observations
    random_state: int = 42

    @T.override
    def split(
        self,
        inputs: schemas.Inputs,
        targets: schemas.Targets,
        groups: Index | None = None,
    ) -> TrainTestSplits:
        check_chronological_inputs(inputs, targets)
        index = np.arange(len(inputs))  # return integer position
        train_index, test_index = model_selection.train_test_split(
            index,
            shuffle=self.shuffle,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        yield train_index, test_index

    @T.override
    def get_n_splits(
        self,
        inputs: schemas.Inputs,
        targets: schemas.Targets,
        groups: Index | None = None,
    ) -> int:
        return 1


class TimeSeriesSplitter(Splitter):
    """Split a dataframe into fixed time series subsets.

    Parameters:
        gap (int): number of observations excluded between train and test.
        n_splits (int): number of split to generate.
        test_size (int): number of observations in each test fold.
    """

    KIND: T.Literal["TimeSeriesSplitter"] = "TimeSeriesSplitter"

    gap: int = pdt.Field(default=0, ge=0)
    n_splits: int = pdt.Field(default=4, ge=2)
    test_size: int = pdt.Field(default=24 * 30 * 2, gt=0)

    @T.override
    def split(
        self,
        inputs: schemas.Inputs,
        targets: schemas.Targets,
        groups: Index | None = None,
    ) -> TrainTestSplits:
        check_chronological_inputs(inputs, targets)
        splitter = model_selection.TimeSeriesSplit(
            n_splits=self.n_splits, test_size=self.test_size, gap=self.gap
        )
        yield from splitter.split(inputs)

    @T.override
    def get_n_splits(
        self,
        inputs: schemas.Inputs,
        targets: schemas.Targets,
        groups: Index | None = None,
    ) -> int:
        return self.n_splits


SplitterKind = TrainTestSplitter | TimeSeriesSplitter
