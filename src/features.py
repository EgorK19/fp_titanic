import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin

__all__ = [
    "AdvancedCabinTransformer",
    "BaselineAgeImputer",
    "BaselineCabinTransformer",
    "ByPclassAgeImputer",
    "ByPclassTitleAgeImputer",
    "BySexAgeImputer",
    "BySexPclassAgeImputer",
    "DtypesTransformer",
    "EmbarkedImputer",
    "FareImputer",
    "GroupTransformer",
    "QuantileBinner",
    "TitleTransformer",
    "ToCategory",
    "ToFloat32Transformer",
]


class DtypesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = X_copy.drop(columns=["PassengerId"], errors="ignore")
        if "Pclass" in X_copy.columns:
            X_copy["Pclass"] = X_copy["Pclass"].astype(str)

        return X_copy


class BaselineAgeImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.global_age_median = None

    def fit(self, X, y=None):
        self.global_age_median = X["Age"].median()
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy["Age"] = X_copy["Age"].fillna(self.global_age_median)
        return X_copy


class ByPclassAgeImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.age_medians = None
        self.global_age_median = None

    def fit(self, X, y=None):
        self.age_medians = X.groupby("Pclass")["Age"].median()
        self.global_age_median = X["Age"].median()
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy["Age"] = X_copy["Age"].fillna(X_copy["Pclass"].map(self.age_medians))

        X_copy["Age"] = X_copy["Age"].fillna(self.global_age_median)

        return X_copy


class BySexAgeImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.medians = None
        self.global_median = None

    def fit(self, X, y=None):
        self.medians = X.groupby("Sex")["Age"].median()
        self.global_median = X["Age"].median()
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy["Age"] = X_copy["Age"].fillna(X_copy["Sex"].map(self.medians))

        X_copy["Age"] = X_copy["Age"].fillna(self.global_median)

        return X_copy


class BySexPclassAgeImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.age_medians = None
        self.global_age_median = None

    def fit(self, X, y=None):
        self.age_medians = X.groupby(["Sex", "Pclass"])["Age"].median().reset_index()
        self.age_medians.rename(columns={"Age": "Group_Median"}, inplace=True)
        self.global_age_median = X["Age"].median()
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        group_key = X_copy.set_index(["Sex", "Pclass"]).index
        group_median = group_key.map(
            self.age_medians.set_index(["Sex", "Pclass"])["Group_Median"]
        )
        X_copy["Age"] = X_copy["Age"].fillna(
            pd.Series(group_median, index=X_copy.index)
        )
        X_copy["Age"] = X_copy["Age"].fillna(self.global_age_median)
        return X_copy


class ByPclassTitleAgeImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.age_medians = None
        self.global_age_median = None

    def fit(self, X, y=None):
        self.age_medians = X.groupby(["Pclass", "Title"])["Age"].median().reset_index()
        self.age_medians.rename(columns={"Age": "Group_Median"}, inplace=True)
        self.global_age_median = X["Age"].median()
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        group_key = X_copy.set_index(["Pclass", "Title"]).index
        group_median = group_key.map(
            self.age_medians.set_index(["Pclass", "Title"])["Group_Median"]
        )
        X_copy["Age"] = X_copy["Age"].fillna(
            pd.Series(group_median, index=X_copy.index)
        )
        X_copy["Age"] = X_copy["Age"].fillna(self.global_age_median)
        return X_copy


class EmbarkedImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.embarked_mode = None

    def fit(self, X, y=None):
        self.embarked_mode = X["Embarked"].mode().iloc[0]
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy["Embarked"] = X_copy["Embarked"].fillna(self.embarked_mode)
        return X_copy


class FareImputer(BaseEstimator, TransformerMixin):
    def __init__(self, impute_nulls=False):
        self.impute_nulls = impute_nulls
        self.global_fare_median = None

    def fit(self, X, y=None):
        self.global_fare_median = X["Fare"].median()
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy["Fare"] = X_copy["Fare"].fillna(self.global_fare_median)

        if self.impute_nulls:
            X_copy["Fare"] = X_copy["Fare"].replace(0, self.global_fare_median)

        return X_copy


# крутое сглаживание лапласа
class GroupTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, smooth=1):
        self.smooth = smooth

    def fit(self, X, y=None):
        X_copy = X.copy()
        y = np.asarray(y)
        X_copy["Target"] = y

        self.survival_mean = X_copy["Target"].mean()
        self.ticket_counts_map = X_copy["Ticket"].value_counts().to_dict()

        ticket_stats = X_copy.groupby(by="Ticket")["Target"].agg(["sum", "count"])
        self.ticket_sum_map = ticket_stats["sum"].to_dict()
        self.ticket_count_map = ticket_stats["count"].to_dict()

        smoothed_survival = (ticket_stats["sum"] + self.smooth * self.survival_mean) / (
            ticket_stats["count"] + self.smooth
        )
        self.ticket_survival_map = smoothed_survival.to_dict()
        return self

    def _base_transform(self, X):
        X_copy = X.copy()
        # фичи без таргета
        X_copy["Ticket_Group_Size"] = (
            X_copy["Ticket"].map(self.ticket_counts_map).fillna(1)
        )
        X_copy["True_Fare"] = X_copy["Fare"] / X_copy["Ticket_Group_Size"]
        X_copy["Log_True_Fare"] = np.log1p(X_copy["True_Fare"])
        return X_copy

    def transform(self, X):
        X_copy = self._base_transform(X)

        X_copy["Ticket_Survival_Rate"] = (
            X_copy["Ticket"].map(self.ticket_survival_map).fillna(self.survival_mean)
        )

        X_copy = X_copy.drop(columns=["Ticket"])  # "Fare", "True_Fare",
        return X_copy

    def fit_transform(self, X, y=None):
        self.fit(X, y)

        X_copy = self._base_transform(X)
        y = np.asarray(y)

        counts = X["Ticket"].map(self.ticket_count_map).values
        sums = X["Ticket"].map(self.ticket_sum_map).values

        loo_counts = counts - 1
        loo_sum = sums - y

        loo_rate = (loo_sum + self.smooth * self.survival_mean) / (
            loo_counts + self.smooth
        )
        loo_rate = np.where(np.isnan(loo_rate), self.survival_mean, loo_rate)

        X_copy["Ticket_Survival_Rate"] = loo_rate
        X_copy = X_copy.drop(columns=["Ticket"])  # "Fare", "True_Fare",
        return X_copy


class BaselineCabinTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy = X_copy.drop(columns="Cabin")
        return X_copy


class AdvancedCabinTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy["Has_Cabin"] = (~X["Cabin"].str[0].isna()).astype(int)
        X_copy["Deck"] = X_copy["Cabin"].str[0].fillna("Unk").replace("T", "A")

        X_copy = X_copy.drop(columns=["Cabin"])
        return X_copy


class TitleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy["Title"] = X_copy["Name"].str.extract(r",\s*([^\s.]+)\s*\.")[0]

        mask = ~X_copy["Title"].isin(["Mr", "Miss", "Mrs", "Master"])

        X_copy.loc[mask, "Title"] = "Extra"

        X_copy = X_copy.drop(columns=["Name"])

        return X_copy


class QuantileBinner(BaseEstimator, TransformerMixin):
    def __init__(self, q=4):
        self.q = q
        self.quantiles_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        else:
            self.feature_names_in_ = np.array(
                [f"x{i}" for i in range(np.shape(X)[1] if len(np.shape(X)) > 1 else 1)],
                dtype=object,
            )

        x_series = pd.Series(np.array(X).ravel()).dropna()

        _, self.quantiles_ = pd.qcut(
            x_series, q=self.q, retbins=True, duplicates="drop"
        )

        return self

    def transform(self, X):
        x_arr = np.array(X).ravel()

        bin_indices = np.digitize(x_arr, self.quantiles_)
        bin_indices = np.clip(bin_indices, 1, len(self.quantiles_) - 1)

        labels = [
            f"({self.quantiles_[i]}, {self.quantiles_[i + 1]}]"
            for i in range(len(self.quantiles_) - 1)
        ]
        string_bins = np.array([labels[idx - 1] for idx in bin_indices], dtype=object)

        string_bins[pd.isna(x_arr)] = "unknown"

        res = pd.DataFrame(string_bins, index=getattr(X, "index", None))

        if hasattr(self, "transform_output_") and self.transform_output_ == "pandas":
            res.columns = self.get_feature_names_out()
            return res

        return res.to_numpy()

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.feature_names_in_
        return np.array(input_features, dtype=object)


class ToCategory(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns)
        self.categories_ = {
            col: pd.Index(X[col].astype("category").cat.categories)
            for col in self.columns
        }
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = pd.Categorical(X_copy[col], categories=self.categories_[col])
        return X_copy


class ToFloat32Transformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):

        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.to_numpy().astype("float32")

        return X.astype("float32")
