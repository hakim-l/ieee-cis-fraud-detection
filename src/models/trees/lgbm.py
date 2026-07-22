from typing import Optional
import joblib

from lightgbm.dask import DaskLGBMClassifier, DaskLGBMRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

from ..base import BaseModel


class LGBMDaskClassifier(BaseModel):
    """Wrapper around lightgbm.dask.DaskLGBMClassifier that implements the BaseModel API.

    Notes:
    - fit accepts an optional dask.distributed.Client as `client` (passed to LightGBM when needed).
    - X and y may be numpy/pandas or dask data structures.
    """

    def __init__(self, **params):
        self.params = params
        self.model = DaskLGBMClassifier(**params)
        self._fitted = False

    def fit(self, X, y, client: Optional[object] = None):
        """Fit the Dask LGBM classifier.

        Accepts client either as first positional argument (LightGBM dask API) or as keyword.
        """
        if client is not None:
            # Try the two common call styles to be robust across LightGBM versions
            try:
                # common docs example: fit(client, X, y)
                self.model.fit(client, X, y)
            except TypeError:
                # alternative: fit(X, y, client=client)
                self.model.fit(X, y, client=client)
        else:
            self.model.fit(X, y)

        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling predict")

        preds = self.model.predict(X)
        # If Dask collection, compute
        if hasattr(preds, "compute"):
            return preds.compute()

        return preds

    def predict_proba(self, X):
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba")

        proba = self.model.predict_proba(X)
        if hasattr(proba, "compute"):
            return proba.compute()
        return proba

    def evaluate(self, X, y, metric: str = "roc_auc"):
        """Evaluate predictions against y.

        Supported metrics: 'roc_auc', 'accuracy'
        """
        # materialize y if dask
        if hasattr(y, "compute"):
            y_true = y.compute()
        else:
            y_true = y

        if metric == "roc_auc":
            proba = self.predict_proba(X)
            if hasattr(proba, "compute"):
                proba = proba.compute()
            # assume binary classification
            score = roc_auc_score(y_true, proba[:, 1])
        elif metric == "accuracy":
            preds = self.predict(X)
            score = accuracy_score(y_true, preds)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        return score

    def save(self, path: str):
        """Persist the wrapped estimator to disk using joblib."""
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str):
        """Load a previously saved estimator and return a wrapper instance."""
        loaded = joblib.load(path)
        inst = cls()
        inst.model = loaded
        inst._fitted = True
        return inst


class LGBMDaskRegressor(BaseModel):
    """Wrapper around lightgbm.dask.DaskLGBMRegressor that implements the BaseModel API."""

    def __init__(self, **params):
        # self.params = params
        self.model = DaskLGBMRegressor(**params)
        self._fitted = False

    def fit(self, X, y, client: Optional[object] = None):
        if client is not None:
            try:
                self.model.fit(client, X, y)
            except TypeError:
                self.model.fit(X, y, client=client)
        else:
            self.model.fit(X, y)

        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling predict")

        preds = self.model.predict(X)
        if hasattr(preds, "compute"):
            return preds.compute()
        return preds

    def evaluate(self, X, y, metric: str = "mse"):
        if hasattr(y, "compute"):
            y_true = y.compute()
        else:
            y_true = y

        preds = self.predict(X)
        if metric == "mse":
            return mean_squared_error(y_true, preds)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def save(self, path: str):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str):
        loaded = joblib.load(path)
        inst = cls()
        inst.model = loaded
        inst._fitted = True
        return inst
