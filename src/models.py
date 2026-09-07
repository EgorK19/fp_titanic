from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (
    AdvancedCabinTransformer,
    ByPclassAgeImputer,
    DtypesTransformer,
    EmbarkedImputer,
    FareImputer,
    GroupTransformer,
    TitleTransformer,
    ToFloat32Transformer,
)


def build_preprocessor_tree_simple() -> ColumnTransformer:
    """бейзлайн"""
    cat_columns = ["Pclass", "Sex", "Embarked"]
    num_columns = ["Age", "SibSp", "Parch", "Fare"]

    cat_transformer = Pipeline(
        [
            ("cat_imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_transformer = Pipeline([("num_imputer", SimpleImputer(strategy="median"))])

    return ColumnTransformer(
        [
            ("cat", cat_transformer, cat_columns),
            ("num", num_transformer, num_columns),
        ]
    )


def build_preprocessor_final_features() -> Pipeline:
    """препроцессор бустингов"""
    return Pipeline(
        [
            ("dtypes", DtypesTransformer()),
            ("title_transformer", TitleTransformer()),
            ("age_imputer", ByPclassAgeImputer()),
            ("embarked_imputer", EmbarkedImputer()),
            ("fare_imputer", FareImputer()),
            ("group_transformer", GroupTransformer(smooth=1e-6)),
            ("cabin_transformer", AdvancedCabinTransformer()),
            (
                "column_transformer",
                ColumnTransformer(
                    [
                        (
                            "cat_ohe",
                            OneHotEncoder(sparse_output=False, drop="first"),
                            ["Pclass", "Sex", "Title", "Has_Cabin"],
                        ),
                        (
                            "num",
                            "passthrough",
                            [
                                "Age",
                                "Ticket_Group_Size",
                                "Fare",
                                "Ticket_Survival_Rate",
                            ],
                        ),
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ).set_output(transform="pandas"),
            ),
        ]
    )


def build_preprocessor_nn_features() -> Pipeline:
    """препроцессор нейросети"""
    return Pipeline(
        [
            ("dtypes", DtypesTransformer()),
            ("title_transformer", TitleTransformer()),
            ("age_imputer", ByPclassAgeImputer()),
            ("embarked_imputer", EmbarkedImputer()),
            ("fare_imputer", FareImputer()),
            ("group_transformer", GroupTransformer(smooth=1e-6)),
            ("cabin_transformer", AdvancedCabinTransformer()),
            (
                "column_transformer",
                ColumnTransformer(
                    [
                        (
                            "cat_ohe",
                            OneHotEncoder(sparse_output=False, drop="first"),
                            ["Pclass", "Sex", "Title", "Deck"],
                        ),
                        (
                            "num",
                            StandardScaler(),
                            [
                                "Age",
                                "SibSp",
                                "Parch",
                                "Log_True_Fare",
                                "Ticket_Survival_Rate",
                            ],
                        ),
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ).set_output(transform="pandas"),
            ),
            ("to_numpy", ToFloat32Transformer()),
        ]
    )


PREPROCESSORS: dict[str, Callable[[], Any]] = {
    "tree_simple": build_preprocessor_tree_simple,
    "final_features": build_preprocessor_final_features,
    "nn_features": build_preprocessor_nn_features,
}

ESTIMATORS: dict[str, Callable[..., Any]] = {
    "lightgbm": lambda seed, params: LGBMClassifier(
        random_state=seed, verbosity=-1, **params
    ),
    "catboost": lambda seed, params: CatBoostClassifier(
        random_state=seed, verbose=0, **params
    ),
}


def _build_mlp3_bn(n_features: int, use_do: bool = False):
    from torch import nn

    class MLP3(nn.Module):
        def __init__(self):
            super().__init__()
            layers = [
                nn.Linear(n_features, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
            ]
            if use_do:
                layers.append(nn.Dropout(0.3))
            layers += [
                nn.Linear(32, 8),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.Linear(8, 1),
            ]
            self.net = nn.Sequential(*layers)

        def forward(self, X):
            return self.net(X)

    return MLP3


NN_ARCHITECTURES: dict[str, Callable[..., Any]] = {
    "mlp3_bn": _build_mlp3_bn,
}


@dataclass
class ModelSpec:
    name: str
    kind: Literal["sklearn", "nn"]
    preprocessor: Any
    estimator: Any = None
    net_builder: Callable | None = None


def build_model_spec(name: str, model_cfg: dict, seed: int) -> ModelSpec:
    preprocessor = PREPROCESSORS[model_cfg["preprocessor"]]()

    if model_cfg["kind"] == "sklearn":
        estimator = ESTIMATORS[model_cfg["estimator"]](
            seed, dict(model_cfg.get("params", {}))
        )
        return ModelSpec(
            name=name, kind="sklearn", preprocessor=preprocessor, estimator=estimator
        )

    if model_cfg["kind"] == "nn":
        import torch
        from skorch import NeuralNetBinaryClassifier
        from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler

        arch_fn = NN_ARCHITECTURES[model_cfg["architecture"]]
        use_do = model_cfg.get("use_do", False)
        p = model_cfg.get("params", {})
        cb_cfg = model_cfg.get("callbacks", {})

        def net_builder(n_features: int):
            module_cls = arch_fn(n_features, use_do=use_do)
            callbacks = [
                (
                    "cb_valid_acc",
                    EpochScoring(
                        scoring="accuracy",
                        name="valid_acc",
                        lower_is_better=False,
                        on_train=False,
                    ),
                ),
                (
                    "lr_scheduler",
                    LRScheduler(
                        policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
                        monitor="valid_acc",
                        mode="max",
                        patience=cb_cfg.get("lr_scheduler_patience", 10),
                        factor=cb_cfg.get("lr_scheduler_factor", 0.5),
                    ),
                ),
                (
                    "early_stopping",
                    EarlyStopping(
                        monitor="valid_acc",
                        patience=cb_cfg.get("early_stopping_patience", 25),
                        lower_is_better=False,
                        load_best=True,
                    ),
                ),
            ]
            return NeuralNetBinaryClassifier(
                module=module_cls,
                max_epochs=p.get("max_epochs", 2000),
                lr=p.get("lr", 3e-4),
                optimizer=torch.optim.Adam,
                batch_size=p.get("batch_size", 32),
                train_split=None,
                callbacks=callbacks,
                iterator_train__shuffle=True,
                iterator_train__drop_last=True,
                verbose=0,
            )

        return ModelSpec(
            name=name, kind="nn", preprocessor=preprocessor, net_builder=net_builder
        )

    raise ValueError(f"Неизвестный kind модели: {model_cfg['kind']!r}")
