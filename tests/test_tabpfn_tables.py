"""No weights required: scientific table contracts and native boundary semantics."""

from types import SimpleNamespace

import numpy as np
import pytest

from biomodals.app.misc.tabpfn.tables import (
    TableFeature,
    TableSchema,
    fit_evaluate_tables,
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
            self.inferred_feature_schema_ = SimpleNamespace(
                features=[
                    SimpleNamespace(modality=SimpleNamespace(value=kind))
                    for kind in ("numerical", "categorical")
                ]
            )

        def predict(self, x, *, output_type):
            assert output_type == "mean"
            calls.append(("predict", len(x)))
            return np.asarray(x[:, 0], dtype=float) * 2

    result, modalities = fit_predict_tables(tables, Estimator(), batch_size=2)
    assert modalities == ("numerical", "categorical")
    assert result.to_dict(as_series=False) == {
        "id": ["x", "y", "z"],
        "predicted_label": [6, 10, 14],
    }
    assert calls == [("fit", 3), ("predict", 2), ("predict", 1)]


@pytest.mark.parametrize("inferred", ["categorical", "numerical"])
def test_high_cardinality_categories_preserve_values_and_verify_native_type(inferred):
    """Never merge numeric spellings or treat a declared category as a quantity."""
    categories = [f"{i:03d}" for i in range(31)] + ["0", "00", ""]
    tables = read_regression_tables(
        (
            "id,group,y\n"
            + "".join(f"m{i},{value},{i}\n" for i, value in enumerate(categories))
        ).encode(),
        b'id,group\na,000\nb,0\nc,031\nd,\ne,""\n',
        TableSchema(
            target="y",
            identifier="id",
            features=(TableFeature(name="group", kind="categorical"),),
        ),
    )

    class Estimator:
        def fit(self, x, y):
            assert x[:, 0].tolist() == categories[:-1] + [None]
            self.inferred_feature_schema_ = SimpleNamespace(
                features=[SimpleNamespace(modality=SimpleNamespace(value=inferred))]
            )

        def predict(self, x, **kwargs):
            assert x[:, 0].tolist() == ["000", "0", "031", None, ""]
            return np.arange(len(x), dtype=float)

    if inferred == "numerical":
        with pytest.raises(ValueError, match="declared categorical"):
            fit_predict_tables(tables, Estimator())
    else:
        result, modalities = fit_predict_tables(tables, Estimator())
        assert modalities == ("categorical",)
        assert result["predicted_label"].to_list() == [0, 1, 2, 3, 4]


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


def test_fold_local_pca_and_final_refit_use_exact_training_rows(monkeypatch):
    """Held-out extremes cannot affect fold projections; final fit restores all rows."""
    from sklearn.decomposition import PCA

    from biomodals.app.misc.tabpfn.tables import RegressionTables

    schema = TableSchema(
        target="y", features=tuple(TableFeature(name=name) for name in ("a", "b", "c"))
    )
    tables = read_regression_tables(
        b"a,b,c,y\n1,2,3,10\n2,4,5,20\n3,5,7,30\n1000,2000,3000,40\n",
        b"a,b,c\n4,6,8\n5,7,9\n",
        schema,
    )
    fitted_pca, fitted_labels = [], []

    class RecordedPCA(PCA):
        def fit_transform(self, x, y=None):
            fitted_pca.append(x.copy())
            return super().fit_transform(x, y)

    class Estimator:
        def fit(self, x, y):
            fitted_labels.append(y.copy())
            self.mean = y.mean()
            self.inferred_feature_schema_ = SimpleNamespace(
                features=[
                    SimpleNamespace(modality=SimpleNamespace(value="numerical"))
                    for _ in range(x.shape[1])
                ]
            )

        def predict(self, x, **kwargs):
            return np.repeat(self.mean, len(x))

    monkeypatch.setattr("sklearn.decomposition.PCA", RecordedPCA)
    predictions, validation, modalities = fit_evaluate_tables(
        tables,
        Estimator,
        validation_folds=((3,), (0,)),
        pca_components=2,
        batch_size=1,
    )
    assert [len(rows) for rows in fitted_pca] == [3, 3, 4]
    assert fitted_pca[0][:, 0].tolist() == [1, 2, 3]
    assert fitted_pca[-1][:, 0].tolist() == [1, 2, 3, 1000]
    assert fitted_labels[-1].tolist() == [10, 20, 30, 40]
    assert predictions["predicted_label"].to_list() == [25, 25]
    assert validation["row_index"].to_list() == [3, 0]
    assert modalities == (("numerical", "numerical"),) * 3
    for bad in (((0, 1, 2),), ((-1,),), ((4,),), ((0, 0),)):
        with pytest.raises(ValueError):
            fit_evaluate_tables(tables, Estimator, validation_folds=bad)
    with pytest.raises(ValueError, match="numeric-only"):
        fit_evaluate_tables(
            RegressionTables(_schema(), tables.training, tables.inference),
            Estimator,
            pca_components=2,
        )
