from pathlib import Path

from omegaconf import OmegaConf

raw_data_dir = Path("d:/vs_projects/fp_titanic/data/raw")

config_dict = {
    "general": {
        "seed": 101,
    },
    "paths": {
        "raw_dir": str(raw_data_dir),
        "train": str(raw_data_dir / "train.csv"),
        "test": str(raw_data_dir / "test.csv"),
        "output": str(Path("d:/vs_projects/fp_titanic/data/external")),
        "logs": str(Path("d:/vs_projects/fp_titanic/models/logs")),
    },
    "run": {
        "models": "all",  # "all" | ["baseline", "best_lgbm", ...]
        "make_submission_for": "best",  # "best" | "all" | "name1,name2"
        "selection_metric": "OOF_acc",
        "save_results_table": True,
        "log_experiments": True,
    },
    "cv": {
        "baseline": {"n_splits": 10, "n_repeats": 1},
        "boosting": {"n_splits": 10, "n_repeats": 5},
        "nn": {"n_splits": 10, "n_repeats": 5},
    },
    "models": {
        "baseline": {
            "kind": "sklearn",
            "preprocessor": "tree_simple",
            "estimator": "lightgbm",
            "params": {"max_depth": 7},
        },
        "best_lgbm": {
            "kind": "sklearn",
            "preprocessor": "final_features",
            "estimator": "lightgbm",
            "params": {
                "n_estimators": 409,
                "max_depth": 5,
                "num_leaves": 104,
                "learning_rate": 0.04755571719306031,
                "min_child_samples": 21,
                "subsample": 0.7279664239579899,
                "colsample_bytree": 0.5322640057787851,
                "reg_alpha": 0.1057163625351844,
                "reg_lambda": 4.628797701922884,
            },
        },
        "best_catboost": {
            "kind": "sklearn",
            "preprocessor": "final_features",
            "estimator": "catboost",
            "params": {
                "iterations": 760,
                "depth": 7,
                "learning_rate": 0.009990468903622285,
                "l2_leaf_reg": 1.3717265878688303,
                "bagging_temperature": 3.583425469372962,
                "border_count": 206,
            },
        },
        "best_nn": {
            "kind": "nn",
            "preprocessor": "nn_features",
            "architecture": "mlp3_bn",
            "use_do": True,
            "params": {
                "lr": 0.0003,
                "batch_size": 32,
                "max_epochs": 2000,
            },
            "callbacks": {
                "lr_scheduler_patience": 10,
                "lr_scheduler_factor": 0.5,
                "early_stopping_patience": 25,
            },
        },
    },
}

cfg = OmegaConf.create(config_dict)
