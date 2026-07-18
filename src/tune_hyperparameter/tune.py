"""Hyperparameter tuning module for IEEE CIS Fraud Detection project.

Provides utilities for tuning model hyperparameters using Optuna and
supports any sklearn-compatible model or BaseModel implementation.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, cross_validate
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from src.config import DEFAULT_RANDOM_STATE
from src.models.base import BaseModel


class OptunaHyperparameterTuner:
    """Hyperparameter tuning for machine learning models using Optuna.
    
    Supports tuning any sklearn-compatible model or BaseModel subclass.
    Uses Bayesian optimization via Tree-structured Parzen Estimator (TPE).
    """

    def __init__(
        self,
        random_state: int = DEFAULT_RANDOM_STATE,
        cv_folds: int = 5,
        n_jobs: int = -1,
        n_trials: int = 100,
        timeout: Optional[int] = None,
    ):
        """Initialize the Optuna-based hyperparameter tuner.
        
        Args:
            random_state: Random seed for reproducibility
            cv_folds: Number of folds for cross-validation
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            n_trials: Maximum number of trials to run
            timeout: Timeout in seconds (None for unlimited)
        """
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        )
        self.best_params = None
        self.best_score = None
        self.study = None
        self.trials_df = None
        logger.info(
            f"Initialized OptunaHyperparameterTuner with {cv_folds}-fold CV "
            f"and {n_trials} max trials"
        )


    def optimize(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        objective_fn: Callable[[optuna.Trial], float],
        direction: str = "maximize",
        show_progress_bar: bool = True,
    ) -> Tuple[Dict[str, Any], float, optuna.Study]:
        """Run Optuna optimization for hyperparameters.
        
        Args:
            model: sklearn-compatible model or BaseModel instance
            X: Features (pandas DataFrame or numpy array)
            y: Target labels (pandas Series or numpy array)
            objective_fn: Callable that takes a Trial and returns a score
            direction: "maximize" or "minimize" the objective
            show_progress_bar: Whether to show optimization progress
            
        Returns:
            Tuple of (best_params, best_score, study_object)
        """
        logger.info(f"Starting Optuna optimization ({direction}) with {self.n_trials} trials")

        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner()

        self.study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
        )

        self.study.optimize(
            objective_fn,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=show_progress_bar,
        )

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        self.trials_df = self.study.trials_dataframe()

        logger.info(f"Optimization complete")
        logger.info(f"Best score: {self.best_score:.6f}")
        logger.info(f"Best parameters: {self.best_params}")

        return self.best_params, self.best_score, self.study

    def optimize_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, Callable],
        scoring: str = "roc_auc",
        direction: str = "maximize",
        show_progress_bar: bool = True,
    ) -> Tuple[Dict[str, Any], float, optuna.Study]:
        """Optimize hyperparameters for any sklearn-compatible or BaseModel.
        
        Works with sklearn models, LightGBM, XGBoost, or any model implementing
        the BaseModel interface.
        
        Args:
            model: Model instance (sklearn, LightGBM, or BaseModel subclass)
            X: Features
            y: Target labels
            param_space: Dict mapping param names to Optuna Trial suggestion methods
                        e.g., {"learning_rate": lambda t: t.suggest_float("lr", 0.01, 0.3)}
            scoring: Scoring metric ("roc_auc", "accuracy", "f1", etc.)
            direction: "maximize" or "minimize"
            show_progress_bar: Whether to show progress
            
        Returns:
            Tuple of (best_params, best_score, study_object)
        """
        def objective(trial: optuna.Trial) -> float:
            # Suggest hyperparameters
            params = {}
            for param_name, suggest_fn in param_space.items():
                params[param_name] = suggest_fn(trial)
            
            # Set parameters on model
            try:
                model.set_params(**params)
            except AttributeError:
                logger.warning(f"Model does not support set_params. Attempting direct attribute assignment.")
                for key, value in params.items():
                    setattr(model, key, value)
            
            # Evaluate using cross-validation
            try:
                cv_results = cross_validate(
                    model,
                    X,
                    y,
                    cv=self.cv,
                    scoring=scoring,
                    n_jobs=self.n_jobs,
                    return_train_score=False,
                )
                score = cv_results[f"test_{scoring}"].mean()
            except Exception as e:
                logger.warning(f"Trial failed with error: {e}. Returning worst score.")
                score = -1.0 if direction == "maximize" else 1e10
            
            return score

        return self.optimize(
            model,
            X,
            y,
            objective,
            direction=direction,
            show_progress_bar=show_progress_bar,
        )

    def get_trials_dataframe(self) -> pd.DataFrame:
        """Get all trials as a pandas DataFrame.
        
        Returns:
            DataFrame with trial information
            
        Raises:
            RuntimeError: If no optimization has been performed
        """
        if self.trials_df is None:
            raise RuntimeError(
                "No optimization results available. Run optimize or optimize_model first."
            )
        return self.trials_df

    def get_top_n_trials(self, n: int = 10) -> pd.DataFrame:
        """Get top N trials from optimization.
        
        Args:
            n: Number of top trials to return
            
        Returns:
            DataFrame with top N trials
        """
        trials_df = self.get_trials_dataframe()
        return trials_df.nlargest(n, "value")[["number", "value", "params"]]

    def get_best_trial(self) -> Dict[str, Any]:
        """Get the best trial details.
        
        Returns:
            Dictionary with best trial information
        """
        if self.study is None:
            raise RuntimeError("No optimization results available.")
        
        return {
            "trial_number": self.study.best_trial.number,
            "score": self.study.best_value,
            "params": self.study.best_params,
        }


