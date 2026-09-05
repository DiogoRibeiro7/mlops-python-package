# Dataframe fingerprints

Format identifier: `bikes.dataframe.v1`.

These SHA-256 fingerprints describe the complete validated numeric/datetime tables used by package jobs. They cover every value and row ID. They do not hash Parquet file bytes, certify source authenticity or recover information discarded by schema validation. In particular, legacy target-component columns are removed and weather values are converted to the model schema’s Float16 representation before hashing.

## Encoding contract

A frame is a payload prefixed by its byte length as an unsigned eight-byte big-endian integer. SHA-256 receives the following framed parts, in order:

1. UTF-8 JSON with keys in this order: `format`, `columns`, `index_name`, `rows`. JSON uses no optional whitespace, unescaped Unicode and the ordered column names.
2. The index array payload.
3. Each column array payload in column order.

Each array payload contains two inner frames: the ASCII NumPy dtype string normalized to little-endian, then the contiguous little-endian array bytes in row order. No sampling, decimal rounding or intermediate per-row hashes are used. Dtype widths, datetime units and signed floating-point zero are significant. Changing only the machine byte order does not change the fingerprint.

Supported arrays are one-dimensional NumPy boolean, integer, unsigned integer, floating-point or datetime arrays with item sizes of 1, 2, 4 or 8 bytes. Values must be finite and nonmissing. Tables must be nonempty, have a single-level index, unique string column names and a string or absent index name. Objects, extension dtypes, time zones and multi-indices are unsupported. Input validation by the bike schemas supplies the supported representation.

A known vector freezes the contract: column `x` with little-endian Int64 values `[1, 2]`, index named `row` with little-endian UInt32 values `[10, 20]`, yields:

```
f8c739f543929fd818345e7da2464454c6c4a842705b9045e7fc071d3cf63e03
```

## Recorded roles

| Job | Roles |
| --- | --- |
| Training | `inputs`, `targets`, `inputs_train`, `targets_train`, `inputs_validation`, `targets_validation` |
| Tuning | `inputs`, `targets` |
| Evaluation | `inputs`, `targets`, plus `reference_inputs` after a supplied reference passes its boundary check |

Training’s validation roles refer to its internal splitter holdout, which the implementation historically names `inputs_test` and `targets_test`. They are distinct from separately supplied final evaluation files. Tuning records its complete input tables here, not individual cross-validation folds.

For each role, the run receives `data.<role>.sha256` and `data.<role>.format` tags plus `provenance/<role>.json`. The artifact contains the format, fingerprint, row count, ordered columns and dtypes, index name and index dtype. This additional metadata contains no raw rows. Existing MLflow lineage logging remains in place.

## Limits and next step

The artifacts and tags remain mutable MLflow records. A hash alone does not prove when a table was observed, whether it was used elsewhere for model selection, or that a model artifact was produced from it. Historical models lack these records and are not backfilled.

Promotion requires the configured MLflow lineage digests and `dataset_sha256` values for all three evaluation roles. Each evaluation hash must match the operator-selected value and have the current format tag. The registered candidate's source run must be active and FINISHED in the configured experiment. Its full `inputs` fingerprint must match `reference_inputs`, with the same encoding format. This binds recorded identities, not actual execution history or tuning exposure.

Expected hashes and the encoding format are saved in `promotion/dataset_sha256.json` before validation, including on rejected attempts. Historical runs without this evidence are rejected. Retrain through the recording workflow rather than backfilling historical tags. Existing lineage and metric checks remain required.
