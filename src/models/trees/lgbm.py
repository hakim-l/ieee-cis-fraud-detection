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

    def __init__(self, model_config: Optional[dict] = None, **params):
        """Initialize LGBM Dask classifier.

        model_config: optional dict containing metadata such as ordered feature list,
        preprocessing info and explicit hyperparameters under key 'hyperparams'.
        """
        self.params = params
        self.model = DaskLGBMClassifier(**params)
        self._fitted = False
        # Ensure model_config stores hyperparams for later reconstruction
        self.model_config = model_config.copy() if model_config else {}
        self.model_config.setdefault("hyperparams", params)

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
        """Persist the wrapped estimator to disk using joblib and write a separate JSON config.

        The model is saved to `path` (e.g., models/lgbm.pkl) and the model_config is written to
        the same path with a `.json` suffix (e.g., models/lgbm.json).

        Additionally, attempt to export an ONNX representation to the same path with `.onnx` suffix
        when possible and when feature metadata is available. Failures during ONNX export are
        non-fatal and only logged.
        """
        from pathlib import Path
        import json

        p = Path(path)
        # save joblib model
        joblib.dump(self.model, str(p))

        # save model_config separately if present
        cfg = getattr(self, "model_config", None)
        if cfg is not None:
            json_path = p.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(cfg, f, indent=2)

        # try export to ONNX if skl2onnx available and feature info exists
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            # Prefer explicit feature list from model_config
            feature_names = None
            if isinstance(cfg, dict):
                feature_names = cfg.get("feature_names") or cfg.get("ordered_features")

            n_features = None
            if feature_names and isinstance(feature_names, (list, tuple)):
                n_features = len(feature_names)
            else:
                # best-effort: try to infer from underlying fitted estimator attributes
                try:
                    # many sklearn-like estimators expose n_features_in_
                    n_features = int(getattr(self.model, "n_features_in_") or getattr(self.model, "_n_features", None))
                except Exception:
                    n_features = None

            if n_features is None:
                print("ONNX export skipped: could not determine number of input features")
            else:
                initial_type = [("input", FloatTensorType([None, n_features]))]
                onnx_model = convert_sklearn(self.model, initial_types=initial_type)
                onnx_path = p.with_suffix(".onnx")
                with open(onnx_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
        except ImportError:
            print("ONNX export skipped: skl2onnx not installed")
        except Exception as exc:
            print(f"ONNX export failed: {exc}")

    @classmethod
    def load(cls, path: str):
        from pathlib import Path
        import json

        p = Path(path)
        loaded = joblib.load(str(p))

        # Support legacy payload where whole dict was saved in joblib
        if isinstance(loaded, dict) and "model" in loaded:
            loaded_model = loaded["model"]
            loaded_cfg = loaded.get("model_config") or {}
        else:
            loaded_model = loaded
            # try separate JSON
            json_path = p.with_suffix(".json")
            if json_path.exists():
                with open(json_path, "r") as f:
                    loaded_cfg = json.load(f)
            else:
                loaded_cfg = {}

        hyperparams = loaded_cfg.get("hyperparams", {}) if isinstance(loaded_cfg, dict) else {}
        inst = cls(model_config=loaded_cfg, **hyperparams) if isinstance(loaded_cfg, dict) else cls()
        inst.model = loaded_model
        inst._fitted = True
        inst.model_config = loaded_cfg or {"hyperparams": getattr(inst, "params", {})}
        return inst


class LGBMDaskRegressor(BaseModel):
    """Wrapper around lightgbm.dask.DaskLGBMRegressor that implements the BaseModel API."""

    def __init__(self, model_config: Optional[dict] = None, **params):
        """Initialize LGBM Dask regressor.

        model_config is optional metadata (see classifier for structure).
        """
        self.params = params
        self.model = DaskLGBMRegressor(**params)
        self._fitted = False
        self.model_config = model_config.copy() if model_config else {}
        self.model_config.setdefault("hyperparams", params)

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
        """Persist the wrapped estimator to disk using joblib and write a separate JSON config.

        The model is saved to `path` (e.g., models/lgbm.pkl) and the model_config is written to
        the same path with a `.json` suffix (e.g., models/lgbm.json).

        Additionally, attempt to export an ONNX representation to the same path with `.onnx` suffix
        when possible and when feature metadata is available. Failures during ONNX export are
        non-fatal and only logged.
        """
        from pathlib import Path
        import json

        p = Path(path)
        # save joblib model
        joblib.dump(self.model, str(p))

        # save model_config separately if present
        cfg = getattr(self, "model_config", None)
        if cfg is not None:
            json_path = p.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(cfg, f, indent=2)

        # try export to ONNX if skl2onnx available and feature info exists
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            # Prefer explicit feature list from model_config
            feature_names = None
            if isinstance(cfg, dict):
                feature_names = cfg.get("feature_names") or cfg.get("ordered_features")

            n_features = None
            if feature_names and isinstance(feature_names, (list, tuple)):
                n_features = len(feature_names)
            else:
                # best-effort: try to infer from underlying fitted estimator attributes
                try:
                    # many sklearn-like estimators expose n_features_in_
                    n_features = int(getattr(self.model, "n_features_in_") or getattr(self.model, "_n_features", None))
                except Exception:
                    n_features = None

            if n_features is None:
                print("ONNX export skipped: could not determine number of input features")
            else:
                initial_type = [("input", FloatTensorType([None, n_features]))]
                onnx_model = convert_sklearn(self.model, initial_types=initial_type)
                onnx_path = p.with_suffix(".onnx")
                with open(onnx_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
        except ImportError:
            print("ONNX export skipped: skl2onnx not installed")
        except Exception as exc:
            print(f"ONNX export failed: {exc}")

    @classmethod
    def load(cls, path: str):
        from pathlib import Path
        import json

        p = Path(path)
        loaded = joblib.load(str(p))
        if isinstance(loaded, dict) and "model" in loaded:
            loaded_model = loaded["model"]
            loaded_cfg = loaded.get("model_config") or {}
        else:
            loaded_model = loaded
            json_path = p.with_suffix(".json")
            if json_path.exists():
                with open(json_path, "r") as f:
                    loaded_cfg = json.load(f)
            else:
                loaded_cfg = {}

        hyperparams = loaded_cfg.get("hyperparams", {}) if isinstance(loaded_cfg, dict) else {}
        inst = cls(model_config=loaded_cfg, **hyperparams) if isinstance(loaded_cfg, dict) else cls()
        inst.model = loaded_model
        inst._fitted = True
        inst.model_config = loaded_cfg or {"hyperparams": getattr(inst, "params", {})}
        return inst
