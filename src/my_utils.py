import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from skorch.dataset import Dataset
from skorch.helper import predefined_split

from src.config import cfg
from src.features import *

__all__ = [
    "build_preproc_pipeline",
    "count_epochs",
    "cv_result",
    "log_experiment",
    "make_submit",
    "nn_cv_result",
    "set_seed",
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


set_seed(cfg.general.seed)


def build_preproc_pipeline(
    age_imputer,
    cabin_transformer,
    group_transformer,
    cat_encoder,
    num_scaler,
    cat_columns,
    num_columns,
):

    preprocessor = Pipeline(
        [
            ("dtypes", DtypesTransformer()),
            ("title_transformer", TitleTransformer()),
            ("age_imputer", age_imputer),
            ("embarked_imputer", EmbarkedImputer()),
            ("fare_imputer", FareImputer()),
            ("group_transformer", group_transformer),
            ("cabin_transformer", cabin_transformer),
            (
                "column_transformer",
                ColumnTransformer(
                    [
                        ("cat", cat_encoder, cat_columns),
                        ("num", num_scaler, num_columns),
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ).set_output(transform="pandas"),
            ),
        ]
    )  # missing data imputation + feature engeneering + scaling/encoding

    return preprocessor


def cv_result(
    model,
    X_train,
    y_train,
    cv_splitter,
    preprocessor,
    fit_params=None,
):
    fit_params = fit_params or {}

    model_pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

    fitted_models = []
    oof_preds = np.zeros(len(X_train), dtype=int)
    tr_scores = []
    val_scores = []

    for train_idx, val_idx in cv_splitter.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        fold_pipe = clone(model_pipe)
        fold_pipe.fit(X_tr, y_tr, **fit_params)

        fold_preds_tr = fold_pipe.predict(X_tr)
        fold_preds_val = fold_pipe.predict(X_val)

        oof_preds[val_idx] = fold_preds_val
        fitted_models.append(fold_pipe)

        tr_scores.append(accuracy_score(y_tr, fold_preds_tr))
        val_scores.append(accuracy_score(y_val, fold_preds_val))

    oof_accuracy = accuracy_score(y_train, oof_preds)

    metrics = pd.DataFrame(
        [
            {
                "TRAIN_acc_MEAN": np.mean(tr_scores),
                "TRAIN_acc_STD": np.std(tr_scores),
                "VAL_acc_MEAN": np.mean(val_scores),
                "VAL_acc_STD": np.std(val_scores),
                "OOF_acc": oof_accuracy,
            }
        ]
    )

    return fitted_models, oof_preds, metrics


def log_experiment(model_name: str, metrics: dict, note: str = ""):
    log_dir = Path(cfg.paths.logs)
    log_file_path = log_dir / "experiments.log"

    log_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "metrics": metrics,
        "note": note,
    }

    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def make_submit(
    model,
    preprocessor,
    X_train,
    y_train,
    X_submit,
    submission_name,
):
    output_dir = Path(cfg.paths.output)
    output_path = output_dir / submission_name

    if output_path.exists():
        stem = output_path.stem
        suffix = output_path.suffix
        counter = 1

        while output_path.exists():
            output_path = output_dir / f"{stem}_{counter}{suffix}"
            counter += 1

    submit_pipe = Pipeline(
        [
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )
    submit_pipe.fit(X_train, y_train)
    submit_preds = submit_pipe.predict(X_submit)

    submission = pd.DataFrame(
        {"PassengerId": X_submit["PassengerId"], "Survived": submit_preds}
    )
    submission.to_csv(output_path, index=False)
    print(f"Файл успешно сохранен: {output_path.name}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def count_epochs(
    skorch_model, preprocessor, X, y, n_repeats=5, base_seed=cfg.general.seed
):
    epochs_estimates = []
    for i in range(n_repeats):
        seed_i = base_seed + i

        X_proxy_tr, X_proxy_val, y_proxy_tr, y_proxy_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed_i
        )

        proxy_preproc_fitted = clone(preprocessor)
        X_proxy_tr_np = proxy_preproc_fitted.fit_transform(X_proxy_tr, y_proxy_tr)
        X_proxy_val_np = proxy_preproc_fitted.transform(X_proxy_val)

        y_proxy_tr_np = y_proxy_tr.to_numpy().astype("float32")
        y_proxy_val_np = y_proxy_val.to_numpy().astype("float32")

        val_dataset = Dataset(X_proxy_val_np, y_proxy_val_np)

        fold_model = clone(skorch_model)

        fold_model.set_params(
            train_split=predefined_split(val_dataset),
            iterator_train__worker_init_fn=seed_worker,
        )

        set_seed(seed_i)
        fold_model.fit(X_proxy_tr_np, y_proxy_tr_np)

        early_stopping_cb = dict(fold_model.callbacks_)["early_stopping"]
        best_epoch = early_stopping_cb.best_epoch_

        epochs_estimates.append(best_epoch)

    return int(np.median(epochs_estimates))


def nn_cv_result(
    skorch_model,
    preprocessor,
    X_train,
    y_train,
    cv_splitter,
    best_epochs=None,
    base_seed=cfg.general.seed,
):
    set_seed(base_seed)
    base_model = clone(skorch_model)

    fold_epochs = []

    fitted_models = []
    oof_preds = np.zeros(len(X_train), dtype=int)
    tr_scores = []
    val_scores = []

    for train_idx, val_idx in cv_splitter.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        fold_preproc = clone(preprocessor)
        X_tr_np = fold_preproc.fit_transform(X_tr, y_tr)
        X_val_np = fold_preproc.transform(X_val)
        y_tr_np = y_tr.to_numpy().astype("float32")
        y_val_np = y_val.to_numpy().astype("float32")

        fold_val_dataset = Dataset(X_val_np, y_val_np)

        fold_model = clone(base_model)
        fold_model.set_params(
            train_split=predefined_split(fold_val_dataset),
            iterator_train__worker_init_fn=seed_worker,
        )
        set_seed(base_seed)
        fold_model.fit(X_tr_np, y_tr_np)

        fold_preds_tr = fold_model.predict(X_tr_np)
        fold_preds_val = fold_model.predict(X_val_np)

        oof_preds[val_idx] = fold_preds_val
        fitted_models.append({"preprocessor": fold_preproc, "model": fold_model})

        tr_scores.append(accuracy_score(y_tr_np, fold_preds_tr))
        val_scores.append(accuracy_score(y_val_np, fold_preds_val))

        early_stopping_cb = dict(fold_model.callbacks_)["early_stopping"]
        fold_best_epoch = early_stopping_cb.best_epoch_
        fold_epochs.append(fold_best_epoch)

    oof_accuracy = accuracy_score(y_train.to_numpy().astype("int"), oof_preds)

    metrics = pd.DataFrame(
        [
            {
                "TRAIN_acc_MEAN": np.mean(tr_scores),
                "TRAIN_acc_STD": np.std(tr_scores),
                "VAL_acc_MEAN": np.mean(val_scores),
                "VAL_acc_STD": np.std(val_scores),
                "OOF_acc": oof_accuracy,
            }
        ]
    )

    return fitted_models, oof_preds, metrics
