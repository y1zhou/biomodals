"""No weights required: scientific table contracts and native boundary semantics."""

import numpy as np
import pytest

from biomodals.app.misc.tabpfn.tables import (
    TableFeature,
    TableSchema,
    fit_predict_tables,
    read_regression_tables,
)


def _schema():
    return TableSchema(
        target="response",
        identifier="id",
        features=(
            TableFeature(name="dose"),
            TableFeature(name="group", kind="categorical"),
        ),
    )


def test_ordered_numeric_categorical_missing_features_fit_once_and_preserve_ids():
    """Column reorder is accepted; missing numeric features are not missing labels."""
    tables = read_regression_tables(
        b"id,group,response,dose\na,001,2,1\nb,002,3,\nc,003,4,5\n",
        b"group,dose,id\n001,3,x\nnew,5,y\n,7,z\n",
        _schema(),
    )
    assert tables.categorical_indices == [1]
    calls = []

    class Estimator:
        def fit(self, x, y):
            calls.append(("fit", len(x)))
            assert x[0].tolist() == [1.0, "001"]
            assert np.isnan(x[1, 0])
            assert y.tolist() == [2, 3, 4]

        def predict(self, x, *, output_type):
            assert output_type == "mean"
            calls.append(("predict", len(x)))
            return np.asarray(x[:, 0], dtype=float) * 2

    result = fit_predict_tables(tables, Estimator(), batch_size=2)
    assert result.to_dict(as_series=False) == {
        "id": ["x", "y", "z"],
        "predicted_label": [6, 10, 14],
    }
    assert calls == [("fit", 3), ("predict", 2), ("predict", 1)]


@pytest.mark.parametrize("bad", ["", "nan", "inf", ">1000", "label"])
def test_invalid_targets_are_not_imputed_or_filtered(bad):
    """A missing or censored target is not a trainable native feature missing value."""
    with pytest.raises(ValueError):
        read_regression_tables(
            f"id,group,response,dose\na,001,{bad},1\n".encode(),
            b"group,dose,id\n001,3,x\n",
            _schema(),
        )


def test_extra_features_identifier_leakage_and_infinite_features_are_rejected():
    """No undeclared column becomes a feature and inference schemas remain exact."""
    with pytest.raises(ValueError, match="exclude target"):
        TableSchema(target="response", features=(TableFeature(name="response"),))
    for inference in (
        b"id,dose,group,extra\nx,1,a,2\n",
        b"id,dose\nx,1\n",
        b"id,dose,group\nx,inf,a\n",
        b"id,dose,group\nx,1,a\nx,2,b\n",
    ):
        with pytest.raises(ValueError):
            read_regression_tables(
                b"id,group,response,dose\na,a,1,1\n", inference, _schema()
            )
