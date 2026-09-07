"""
Точка входа пайплайна.
Не до коднца разобарлся с назначением, поэтому часть с config, main, modules получилась довольно вымученной
Я вроде нормально в ipynb файлах все крутил и проверял, этот main файл скорее нужен для финального решения или просто для удобства проверки?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parent))

from src.config import cfg
from src.models import build_model_spec
from src.my_utils import cv_result, make_submit, set_seed

try:
    from src.my_utils import nn_cv_result
except ImportError:
    nn_cv_result = None

try:
    from src.my_utils import log_experiment
except ImportError:
    log_experiment = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Titanic: baseline + бустинги + NN")
    parser.add_argument(
        "--models", nargs="+", default=None, help="Переопределить run.models"
    )
    parser.add_argument(
        "--submit-for", type=str, default=None, help="best | all | имя1,имя2"
    )
    return parser.parse_args()


def get_models_to_run(cfg) -> list[str]:
    if cfg.run.models == "all":
        return list(cfg.models.keys())
    return list(cfg.run.models)


def get_cv_splitter(cfg, kind: str, seed: int):
    from sklearn.model_selection import RepeatedStratifiedKFold

    section = (
        cfg.cv.baseline if kind == "baseline" else cfg.cv.get(kind, cfg.cv.boosting)
    )
    return RepeatedStratifiedKFold(
        n_splits=section.n_splits, n_repeats=section.n_repeats, random_state=seed
    )


def run_one_model(name: str, spec, X, y, cfg, seed: int) -> tuple[dict, object]:
    """возвращает (строка метрик, "исполнитель" для последующего сабмита)"""
    cv_kind = (
        "baseline"
        if name == "baseline"
        else ("nn" if spec.kind == "nn" else "boosting")
    )
    cv_splitter = get_cv_splitter(cfg, cv_kind, seed)

    if spec.kind == "sklearn":
        _, _, metrics = cv_result(spec.estimator, X, y, cv_splitter, spec.preprocessor)
        row = metrics.iloc[0].to_dict()
        row["model"] = name
        return row, spec  # spec = estimator + preprocessor

    # kind == "nn"
    if nn_cv_result is None:
        raise RuntimeError(
            "src.my_utils.nn_cv_result не найден - проверьте импорт/сигнатуру"
        )

    n_features = spec.preprocessor.fit_transform(X, y).shape[1]
    net = spec.net_builder(n_features)
    fitted_models, _, metrics = nn_cv_result(
        net, spec.preprocessor, X, y, cv_splitter, base_seed=seed
    )
    row = metrics.iloc[0].to_dict()
    row["model"] = name
    return row, fitted_models  # модельки


def make_submission(name: str, executor, X, y, df_submit, cfg) -> None:
    out_dir = Path(cfg.paths.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{name}_submission.csv"

    if isinstance(executor, list):  # просто усреднение по фолдам
        import numpy as np

        test_preds_list = []
        for fold in executor:
            preproc = fold["preprocessor"]
            model = fold["model"]
            X_test = preproc.transform(df_submit)
            test_preds_list.append(model.predict_proba(X_test))
        final_probs = np.mean(test_preds_list, axis=0)
        final_preds = np.argmax(final_probs, axis=1)
        submission = pd.DataFrame(
            {"PassengerId": df_submit["PassengerId"], "Survived": final_preds}
        )
        submission.to_csv(out_dir / filename, index=False)
        print(f"[submit] {name}: сохранено в {out_dir / filename}")
    else:
        # make_submit
        make_submit(
            executor.estimator, executor.preprocessor, X, y, df_submit, filename
        )


def main() -> None:
    args = parse_args()

    if args.models:
        cfg.run.models = args.models
    if args.submit_for:
        cfg.run.make_submission_for = args.submit_for

    seed = cfg.general.seed
    set_seed(seed)

    df_train = pd.read_csv(Path(cfg.paths.train))
    df_submit = pd.read_csv(Path(cfg.paths.test))
    X, y = df_train.drop(columns=["Survived"]), df_train["Survived"]

    model_names = get_models_to_run(cfg)
    results, executors = [], {}

    for name in model_names:
        model_cfg = OmegaConf.to_container(cfg.models[name], resolve=True)
        spec = build_model_spec(name, model_cfg, seed)

        print(f"\n=== {name} ===")
        row, executor = run_one_model(name, spec, X, y, cfg, seed)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

        results.append(row)
        executors[name] = executor

        if cfg.run.log_experiments and log_experiment is not None:
            log_experiment(model_name=name, metrics=row, note="итоговый прогон main.py")

    results_df = pd.DataFrame(results).set_index("model")
    metric = cfg.run.selection_metric
    results_df = results_df.sort_values(metric, ascending=False)

    print("\n=== Итоговая таблица результатов ===")
    print(results_df)

    if cfg.run.save_results_table:
        out_dir = Path(cfg.paths.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_dir / "results_table.csv")
        print(f"\n[table] сохранено в {out_dir / 'results_table.csv'}")

    best_name = results_df.index[0]
    print(
        f"\nЛучшая модель по {metric}: {best_name} ({results_df.loc[best_name, metric]:.4f})"
    )

    submit_policy = cfg.run.make_submission_for
    if submit_policy == "best":
        targets = [best_name]
    elif submit_policy == "all":
        targets = model_names
    else:
        targets = [s.strip() for s in str(submit_policy).split(",")]

    for name in targets:
        make_submission(name, executors[name], X, y, df_submit, cfg)


if __name__ == "__main__":
    main()
