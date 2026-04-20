from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


@dataclass
class FittedModel:
    name: str
    estimator: Any
    features: list[str]
    target: str


def select_numeric_features(df: pd.DataFrame, exclude_cols: list[str]) -> list[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in exclude_cols]


def fit_ridge(
    train_df: pd.DataFrame,
    features: list[str],
    target: str,
    alpha: float = 1.0,
) -> FittedModel:
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )

    pipe.fit(train_df[features], train_df[target])

    return FittedModel(
        name="ridge",
        estimator=pipe,
        features=features,
        target=target,
    )


def fit_lgbm(
    train_df: pd.DataFrame,
    features: list[str],
    target: str,
    random_state: int = 42,
    **params,
) -> FittedModel:
    if lgb is None:
        raise ImportError("lightgbm is not installed.")

    default_params = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        min_child_samples=50,
        objective="regression",
        random_state=random_state,
        n_jobs=-1,
    )
    default_params.update(params)

    model = lgb.LGBMRegressor(**default_params)
    model.fit(train_df[features], train_df[target])

    return FittedModel(
        name="lightgbm",
        estimator=model,
        features=features,
        target=target,
    )


def predict(model: FittedModel, df: pd.DataFrame) -> np.ndarray:
    return model.estimator.predict(df[model.features])