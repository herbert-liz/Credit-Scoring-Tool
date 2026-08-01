"""Integrated credit scoring pipeline."""

from copy import deepcopy

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from creditScoring.config.default_config import PipelineConfig, get_default_config
from creditScoring.feature_selection.correlation import remove_correlated_features
from creditScoring.feature_selection.iv_selection import IVFeatureSelector
from creditScoring.models import (
    LogisticRegressionModel,
    NeuralNetworkModel,
    RandomForestModel,
    XGBoostModel,
)
from creditScoring.preprocessing import BinningTransformer, CategoricalEncoder, WOETransformer, handle_missing_values


class CreditScoringPipeline:
    """End-to-end credit scoring pipeline with configurable modules."""

    def __init__(self, config=None):
        if config is None:
            self.config = get_default_config()
        elif isinstance(config, PipelineConfig):
            self.config = config.to_dict()
        else:
            self.config = deepcopy(config)

        self.encoder = CategoricalEncoder(method="target")
        self.binner = BinningTransformer(
            method=self.config["binning"]["method"],
            n_bins=self.config["binning"]["n_bins"],
            min_bin_pct=self.config["binning"].get("min_bin_pct", 0.05),
        )
        self.woe = WOETransformer()
        self.standard_scaler = StandardScaler()
        self.iv_selector = IVFeatureSelector(min_iv=self.config["feature_selection"]["min_iv"])
        self.model = self._create_model(self.config.get("model_type", "logistic"))
        self.selected_features_: list[str] = []
        self._fitted = False

    def _create_model(self, model_type: str):
        if model_type == "logistic":
            return LogisticRegressionModel(random_state=self.config.get("random_state", 42))
        if model_type == "random_forest":
            return RandomForestModel(random_state=self.config.get("random_state", 42))
        if model_type == "neural_network":
            return NeuralNetworkModel(random_state=self.config.get("random_state", 42))
        if model_type == "xgboost":
            return XGBoostModel(random_state=self.config.get("random_state", 42))
        raise ValueError(
            f"Unsupported model_type: {model_type}. Supported types: logistic, random_forest, neural_network, xgboost"
        )

    def preprocess(self, X: pd.DataFrame, y: pd.Series | None = None, fit: bool = False) -> pd.DataFrame:
        data = handle_missing_values(
            X,
            numeric_strategy=self.config["missing"]["numeric_strategy"],
            categorical_strategy=self.config["missing"]["categorical_strategy"],
            numeric_fill_value=self.config["missing"]["fill_value"],
            categorical_fill_value=self.config["missing"]["categorical_fill_value"],
        )
        if fit:
            self.encoder.fit(data, y)
        data = self.encoder.transform(data)

        model_type = self.config.get("model_type", "logistic")

        if model_type == "logistic":
            # Binning + WOE transformation for logistic regression
            if fit:
                self.binner.fit(data, y)
            if self.binner.method in {"supervised", "monotonic"} and fit:
                data = self.binner.transform(data, y)
            else:
                data = self.binner.transform(data)

            data = data.astype(str)
            if fit:
                self.woe.fit(data, y)
            data = self.woe.transform(data)
        elif model_type == "neural_network":
            # StandardScaler for neural network
            if fit:
                self.standard_scaler.fit(data)
            data = pd.DataFrame(
                self.standard_scaler.transform(data),
                columns=data.columns,
                index=data.index,
            )
        # random_forest and xgboost: no additional transformation needed

        return data

    def select_features(self, X: pd.DataFrame, y: pd.Series, fit: bool = False) -> pd.DataFrame:
        data = X.copy()
        if fit:
            self.iv_selector.fit(data, y)
            ranked = self.iv_selector.get_ranking()
            selected = remove_correlated_features(
                data,
                ranked,
                threshold=self.config["feature_selection"]["max_correlation"],
                method=self.config["feature_selection"]["correlation_method"],
            )
            self.selected_features_ = selected or self.iv_selector.selected_features_ or data.columns.tolist()
        return data[self.selected_features_].copy()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        Xp = self.preprocess(X, y=y, fit=True)
        Xs = self.select_features(Xp, y, fit=True)
        self.model.fit(Xs, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame):
        self._check_fitted()
        Xp = self.preprocess(X, fit=False)
        Xs = self.select_features(Xp, y=None, fit=False)
        return self.model.predict(Xs)

    def predict_proba(self, X: pd.DataFrame):
        self._check_fitted()
        Xp = self.preprocess(X, fit=False)
        Xs = self.select_features(Xp, y=None, fit=False)
        return self.model.predict_proba(Xs)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("Loaded object is not a CreditScoringPipeline instance")
        return loaded

    def _check_fitted(self):
        if not self._fitted:
            raise ValueError("Pipeline must be fitted before prediction")
