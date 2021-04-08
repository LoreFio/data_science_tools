import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.data_tools import unique_is_bool


class NanScaler(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols, max_q=0.95, min_q=0.05):
        """Scaler equivalent to sklearn RobustScaler with the ability to deal with NaN values

        :param numeric_cols: columns containing numeric values to be scaled
        :type numeric_cols: list
        :param max_q: Maximum quantile. Default is 0.75
        :type max_q: float
        :param min_q: Minimum quantile. Default is 0.25
        :type min_q: float
        """
        self.max_q = max_q
        self.min_q = min_q
        self.mode_df = {}
        self.iqr_df = {}
        self.numeric_cols = numeric_cols or []
        self.bool_list = []

    def fit(self, X, y=None):
        """Computes mode and inter-quantile range for all columns in a database

        :param X: pd.DataFrame or pd.Series.
        :type X: pd.DataFrame
        :param y: Not used.
        :type y: pd.DataFrame
        :return: self
        :rtype: NanScaler
        """
        if isinstance(X, pd.DataFrame):
            self.numeric_cols = list(set(X.columns) & set(self.numeric_cols))
        elif isinstance(X, pd.Series):
            self.numeric_cols = []
        else:
            TypeError("X must be a pd.DataFrame or a pd.Series")
        for col in self.numeric_cols:
            if unique_is_bool(X[col].unique()):
                self.bool_list.append(col)
            invalid_mask = X[col].isin([-np.inf, np.nan, np.inf])
            quantiles = X[col][~invalid_mask].quantile([self.min_q, 0.5, self.max_q])
            self.mode_df[col] = quantiles.iloc[1]
            self.iqr_df[col] = quantiles.iloc[2] - quantiles.iloc[0]
            if np.isclose(self.iqr_df[col], 0., rtol=1e-6):
                self.iqr_df[col] = 1
        return self

    def transform(self, X):
        """Scales input X

        :param X: pd.DataFrame or pd.Series.
        :type X: pd.DataFrame
        :return: scaled values
        :rtype: pd.DataFrame
        """
        X = X.copy()
        for col in self.numeric_cols:
            if col not in self.bool_list:
                if isinstance(X, pd.Series):
                    X = (X - self.mode_df[col]) / self.iqr_df[col]
                elif isinstance(X, pd.DataFrame):
                    X[col] = (X[col] - self.mode_df[col]) / self.iqr_df[col]
                else:
                    TypeError("X must be a pd.DataFrame or a pd.Series")
        return X

    def inverse_transform(self, X):
        """Unscales input X

        :param X: pd.DataFrame or pd.Series.
        :type X: pd.DataFrame
        :return: unscaled values
        :rtype: pd.DataFrame
        """
        X = X.copy()
        for col in self.numeric_cols:
            if col not in self.bool_list:
                if isinstance(X, pd.Series):
                    X.loc[col] = X[col] * self.iqr_df[col] + self.mode_df[col]
                elif isinstance(X, pd.DataFrame):
                    X[col] = X[col].apply(lambda x: x * self.iqr_df[col] + self.mode_df[col])
                else:
                    TypeError("X must be a pd.DataFrame or a pd.Series")
        return X
