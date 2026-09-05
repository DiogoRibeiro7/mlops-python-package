# %% IMPORTS

import pandas as pd
import pandera as pa
import pytest

from bikes.core import models, schemas
from bikes.io import datasets

# %% SCHEMAS


def test_inputs_schema(inputs_reader: datasets.Reader) -> None:
    # given
    schema = schemas.InputsSchema
    # when
    data = inputs_reader.read()
    # then
    assert schema.check(data) is not None, "Inputs data should be valid!"


def test_targets_schema(targets_reader: datasets.Reader) -> None:
    # given
    schema = schemas.TargetsSchema
    # when
    data = targets_reader.read()
    # then
    assert schema.check(data) is not None, "Targets data should be valid!"


def test_outputs_schema(outputs_reader: datasets.Reader) -> None:
    # given
    schema = schemas.OutputsSchema
    # when
    data = outputs_reader.read()
    # then
    assert schema.check(data) is not None, "Outputs data should be valid!"


def test_shap_values_schema(
    model: models.Model,
    train_test_sets: tuple[schemas.Inputs, schemas.Targets, schemas.Inputs, schemas.Targets],
) -> None:
    # given
    schema = schemas.SHAPValuesSchema
    _, _, inputs_test, _ = train_test_sets
    # when
    data = model.explain_samples(inputs=inputs_test)
    # then
    assert schema.check(data) is not None, "SHAP values data should be valid!"


def test_feature_importances_schema(model: models.Model) -> None:
    # given
    schema = schemas.FeatureImportancesSchema
    # when
    data = model.explain_model()
    # then
    assert schema.check(data) is not None, "Feature importance data should be valid!"


@pytest.mark.parametrize("column", ["casual", "registered"])
def test_inputs_schema_removes_legacy_counts(inputs: schemas.Inputs, column: str) -> None:
    legacy = inputs.assign(**{column: 123})
    checked = schemas.InputsSchema.check(legacy)
    pd.testing.assert_frame_equal(checked, inputs)
    assert column in legacy.columns, "Validation must not mutate the caller!"


@pytest.mark.parametrize("column", ["cnt", "unexpected"])
def test_inputs_schema_rejects_other_extra_columns(inputs: schemas.Inputs, column: str) -> None:
    with pytest.raises(pa.errors.SchemaError):
        schemas.InputsSchema.check(inputs.assign(**{column: 123}))
