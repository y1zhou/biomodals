"""Explicit numerical/categorical schemas at the native estimator boundary."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_TABLE_BYTES = 128 * 1024 * 1024
MAX_TRAIN_ROWS = 10_000
MAX_PREDICT_ROWS = 100_000
MAX_FEATURES = 20_000
MAX_TABLE_CELLS = 400_000_000
MAX_PARQUET_BYTES = 2 * 1024**3


class TableFeature(BaseModel):
    """A named native feature, never inferred from an identifier column."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str = Field(min_length=1, max_length=200)
    kind: Literal["numeric", "categorical"] = "numeric"


class TableSchema(BaseModel):
    """The same ordered features apply to training and inference tables."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    target: str = Field(min_length=1, max_length=200)
    identifier: str | None = Field(default=None, min_length=1, max_length=200)
    features: tuple[TableFeature, ...] = Field(min_length=1, max_length=MAX_FEATURES)

    @model_validator(mode="after")
    def disjoint_names(self) -> TableSchema:
        """Exclude target/ID leakage and ambiguous duplicate features."""
        names = [feature.name for feature in self.features]
        if (
            self.target in names
            or self.identifier in names
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "Feature names must be unique and exclude target/identifier"
            )
        if self.target == self.identifier:
            raise ValueError("Target and identifier must differ")
        if self.identifier == "predicted_label":
            raise ValueError(
                "Identifier cannot use the reserved predicted_label output name"
            )
        return self


@dataclass(frozen=True)
class RegressionTables:
    """Validated Polars tables; conversion is deferred to native fit/predict."""

    schema: TableSchema
    training: pl.DataFrame
    inference: pl.DataFrame

    @property
    def feature_names(self) -> list[str]:
        """Retain the explicitly declared, shared feature order."""
        return [feature.name for feature in self.schema.features]

    @property
    def categorical_indices(self) -> list[int]:
        """Return native categorical feature indices, not an external one-hot map."""
        return [
            i
            for i, feature in enumerate(self.schema.features)
            if feature.kind == "categorical"
        ]


def _read_table(content: bytes, schema: TableSchema, *, training: bool) -> pl.DataFrame:
    if not 0 < len(content) <= MAX_TABLE_BYTES:
        raise ValueError("Table must be nonempty and at most 128 MiB")
    limit = MAX_TRAIN_ROWS if training else MAX_PREDICT_ROWS
    try:
        frame = pl.read_csv(BytesIO(content), infer_schema=False, n_rows=limit + 1)
    except pl.exceptions.PolarsError as exc:
        raise ValueError("Invalid UTF-8 CSV table") from exc
    return validate_table(frame, schema, training=training)


def validate_table(
    frame: pl.DataFrame, schema: TableSchema, *, training: bool
) -> pl.DataFrame:
    """Validate the same schema for standalone CSV and workflow feature tables."""
    limit = MAX_TRAIN_ROWS if training else MAX_PREDICT_ROWS
    columns = [feature.name for feature in schema.features]
    if training:
        columns.append(schema.target)
    if schema.identifier is not None:
        columns.append(schema.identifier)
    if set(frame.columns) != set(columns):
        raise ValueError(
            "Table columns must exactly match declared features, target (training only), and identifier"
        )
    minimum = 2 if training else 1
    if not minimum <= frame.height <= limit:
        raise ValueError(f"Table must contain between {minimum} and {limit} rows")
    if frame.height * len(schema.features) > MAX_TABLE_CELLS:
        raise ValueError("Table exceeds the feature-cell resource ceiling")
    text_columns = [f.name for f in schema.features if f.kind == "categorical"]
    if schema.identifier is not None:
        text_columns.append(schema.identifier)
    if any(frame.schema[name] != pl.String for name in text_columns):
        raise ValueError("Categorical features and identifiers must be strings")
    numeric = [feature.name for feature in schema.features if feature.kind == "numeric"]
    if training:
        numeric.append(schema.target)
    converted = frame.with_columns(
        pl.col(name).cast(pl.Float64, strict=False) for name in numeric
    )
    for name in numeric:
        if (frame[name].is_not_null() & converted[name].is_null()).any() or converted[
            name
        ].is_infinite().any():
            raise ValueError(f"Column {name!r} contains a nonnumeric value or infinity")
    if training and not converted[schema.target].is_finite().fill_null(False).all():
        raise ValueError("Every target must be a finite numeric measurement")
    if schema.identifier is not None:
        identifiers = converted[schema.identifier]
        if (
            identifiers.is_null().any()
            or identifiers.str.strip_chars().eq("").any()
            or identifiers.n_unique() != len(identifiers)
        ):
            raise ValueError("Table identifiers must be nonempty and unique")
    return converted.select(columns)


def read_table_file(path: Path, schema: TableSchema, *, training: bool) -> pl.DataFrame:
    """Read only bounded local CSV/Parquet, never arbitrary native model files."""
    from biomodals.helper.artifacts import read_bounded_file_bytes

    if path.suffix == ".csv":
        return _read_table(
            read_bounded_file_bytes(
                path, field_name="table", max_bytes=MAX_TABLE_BYTES
            ),
            schema,
            training=training,
        )
    if path.suffix != ".parquet" or not 0 < path.stat().st_size <= MAX_PARQUET_BYTES:
        raise ValueError("Expected a bounded CSV or Parquet feature table")
    limit = MAX_TRAIN_ROWS if training else MAX_PREDICT_ROWS
    lazy = pl.scan_parquet(path)
    if len(lazy.collect_schema()) > MAX_FEATURES + 2:
        raise ValueError("Too many table columns")
    count = lazy.select(pl.len()).collect().item()
    if count > limit or count * len(schema.features) > MAX_TABLE_CELLS:
        raise ValueError("Table exceeds its row or feature-cell resource ceiling")
    return validate_table(lazy.collect(), schema, training=training)


def read_regression_tables(
    training_csv: bytes, inference_csv: bytes, schema: TableSchema
) -> RegressionTables:
    """Parse using all rows, preserve category strings and reorder known features."""
    return RegressionTables(
        schema,
        _read_table(training_csv, schema, training=True),
        _read_table(inference_csv, schema, training=False),
    )


def fit_predict_tables(
    tables: RegressionTables,
    estimator,
    *,
    batch_size: int = 256,
    feature_transform=None,
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """Fit once on all rows, convert only at native boundaries, and preserve IDs."""
    import numpy as np

    if not 1 <= batch_size <= 4096:
        raise ValueError("Prediction batch size must be between 1 and 4096")
    training = tables.training.select(tables.feature_names).to_numpy()
    if feature_transform is not None:
        training = feature_transform.fit_transform(training)
    estimator.fit(training, tables.training[tables.schema.target].to_numpy())
    modalities = tuple(
        feature.modality.value
        for feature in estimator.inferred_feature_schema_.features
    )
    if len(modalities) != training.shape[1] or any(
        modalities[index] != "categorical" for index in tables.categorical_indices
    ):
        raise ValueError(
            "Native feature modalities changed a declared categorical feature or width"
        )
    predictions = []
    for batch in tables.inference.iter_slices(batch_size):
        features = batch.select(tables.feature_names).to_numpy()
        if feature_transform is not None:
            features = feature_transform.transform(features)
        predicted = np.asarray(
            estimator.predict(features, output_type="mean"),
            dtype=np.float64,
        )
        if predicted.shape != (len(batch),) or not np.isfinite(predicted).all():
            raise ValueError("Native estimator returned invalid predictions")
        predictions.append(pl.Series("predicted_label", predicted))
    labels = pl.concat(predictions)
    identifiers = (
        tables.inference.select(tables.schema.identifier)
        if tables.schema.identifier is not None
        else pl.DataFrame({"row_index": range(tables.inference.height)})
    )
    return identifiers.with_columns(labels), modalities


def fit_evaluate_tables(
    tables: RegressionTables,
    estimator_factory,
    *,
    batch_size: int = 256,
    validation_folds: tuple[tuple[int, ...], ...] = (),
    pca_components: int | None = None,
    seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame, tuple[tuple[str, ...], ...]]:
    """Fresh fold-local preprocessing/models, then an unconditional all-data refit.

    This generic boundary receives explicit holdout indices. The caller owns the
    scientific split policy; this app never invents random validation rows.
    """
    if pca_components is not None and (
        pca_components < 1 or tables.categorical_indices
    ):
        raise ValueError(
            "PCA requires numeric-only features and a positive component bound"
        )
    if len(validation_folds) > 5:
        raise ValueError("At most five validation folds are supported")
    for indices in validation_folds:
        if (
            not indices
            or len(set(indices)) != len(indices)
            or min(indices) < 0
            or max(indices) >= tables.training.height
        ):
            raise ValueError("Invalid held-out row indices")
        if tables.training.height - len(indices) < 2:
            raise ValueError(
                "Every validation fold must retain at least two training rows"
            )
    modalities = []

    def predict(subset):
        transform = None
        width = len(subset.feature_names)
        if pca_components is not None and width > pca_components:
            from sklearn.decomposition import PCA

            width = min(pca_components, subset.training.height - 1)
            transform = PCA(
                n_components=width, svd_solver="randomized", random_state=seed
            )
        predictions, inferred = fit_predict_tables(
            subset,
            estimator_factory(),
            batch_size=batch_size,
            feature_transform=transform,
        )
        modalities.append(inferred)
        return predictions

    records = []
    for fold_index, indices in enumerate(validation_folds):
        held = set(indices)
        training_indices = [i for i in range(tables.training.height) if i not in held]
        inference_columns = tables.feature_names + (
            [] if tables.schema.identifier is None else [tables.schema.identifier]
        )
        subset = RegressionTables(
            tables.schema,
            tables.training[training_indices],
            tables.training[list(indices)].select(inference_columns),
        )
        values = predict(subset)
        records.append(
            pl.DataFrame({
                "fold_index": pl.Series([fold_index] * len(indices), dtype=pl.UInt32),
                "row_index": pl.Series(indices, dtype=pl.UInt32),
                "predicted_label": values["predicted_label"],
            })
        )
    validation = (
        pl.concat(records)
        if records
        else pl.DataFrame(
            schema={
                "fold_index": pl.UInt32,
                "row_index": pl.UInt32,
                "predicted_label": pl.Float64,
            }
        )
    )
    predicted = predict(tables)
    return predicted, validation, tuple(modalities)
