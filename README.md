# Titanic - Final Project

Бинарная классификация выживаемости пассажиров «Титаника» ([Kaggle: Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)). 

### Результат

| Модель                               | OOF Accuracy   | Kaggle LB   |
| ------------------------------------ | -------------- | ----------- |
| baseline (LightGBM, без фичей)       | 0.838352 (val) | 0.77033     |
| **best_lgbm** (LightGBM, Optuna)     | **0.8575**     | **0.78468** |
| best_catboost (CatBoost, Optuna)     | 0.8575         | 0.77751     |
| best_nn (MLP3 + BatchNorm, ensemble) | 0.8485         | 0.78229     |

Итоговая модель - **LightGBM** (`best_lgbm`): при равном OOF-score с CatBoost показала более высокий скор на публичном лидерборде Kaggle. Подробности экспериментов и обоснование выбора фичей - в `notebooks/`.

### Установка

Проект использует [uv](https://docs.astral.sh/uv/).

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Установка зависимостей + Python + venv:

```bash
uv sync
```

### Данные

Датасет не входит в репозиторий

1. `train.csv` и `test.csv` взяты со страницы соревнования [Kaggle: Titanic](https://www.kaggle.com/competitions/titanic/data).
2. Я хранил оба файла в `data/raw/`.

### Сonfig

Все настройки - в одном месте, `src/config.py`:

- `paths` - пути к данным, выходам и логам. **Если запускаете не на Windows или из другой директории - поправьте `raw_data_dir` под свою систему** (по умолчанию там абсолютный путь вида `d:/vs_projects/fp_titanic/...`).
- `models` - реестр моделей (`baseline`, `best_lgbm`, `best_catboost`, `best_nn`) с препроцессором, эстиматором и гиперпараметрами каждой.
- `run` - какие модели гонять (`models: "all"` либо список), для кого делать сабмит (`make_submission_for: "best" | "all" | список`), по какой метрике выбирать победителя (`selection_metric`, по умолчанию `OOF_acc`).
- `cv` - параметры кросс-валидации отдельно для baseline / бустингов / NN.

### Запуск

```bash
# прогнать все модели из run.models конфига
uv run python main.py

# прогнать только конкретные модели
uv run python main.py --models baseline best_lgbm

# сделать сабмит для всех моделей, а не только для победителя
uv run python main.py --submit-for all
```

### Логика `main.py`

1. Берёт `cfg` из `src.config` и грузит `train.csv` / `test.csv`.
2. Для каждой модели из `src/models.py` собирает preprocessor + estimator и прогоняет по CV (`cv_result` - для sklearn-совместимых моделей, `nn_cv_result` - для нейросетей).
3. Собирает единую таблицу метрик по всем моделям.
4. Выбирает победителя по `run.selection_metric`.
5. Делает сабмит для победителя (или всех/списка).

### Выход

Всё сохраняется в `data/external/`:

- `<model_name>_submission.csv` - сабмит для Kaggle (`PassengerId, Survived`) по каждой запрошенной модели.
- `results_table.csv` - сводная таблица метрик по всем прогнанным моделям, отсортированная по `selection_metric`.

Лог каждого запуска пишется в `models/logs/experiments.log`.

### Ноутбуки

`notebooks/` - черновики и история экспериментов:

- `0.0_draft.ipynb` - свалка
- `1.0_eda.ipynb` - разведочный анализ данных
- `2.0_baseline.ipynb` - простой baseline
- `3.0_modeling.ipynb` - эксперименты с не-нейросетевыми моделями, отбор фичей
- `4.0_final_boosting.ipynb` - Optuna и обучение финальных бустингов
- `5.0_nn.ipynb` - эксперименты с нейросетью, финальная ансамблевая архитектура MLP3 + BatchNorm

### Структура проекта

```text
.
├── data/
│   ├── external/               # сабмиты и таблица метрик (выход main.py)
│   ├── interim/                # не использовалось
│   ├── processed/              # не использовалось
│   └── raw/                    # train.csv / test.csv
│
├── models/
│   └── logs/
│       └── experiments.log     # лог результатов и метрик всех запусков
│
├── notebooks/                  # jupyter-ноутбуки: EDA и история экспериментов
│   ├── 0.0_draft.ipynb
│   ├── 1.0_eda.ipynb
│   ├── 2.0_baseline.ipynb
│   ├── 3.0_modeling.ipynb
│   ├── 4.0_final_boosting.ipynb
│   └── 5.0_nn.ipynb
│
├── src/
│   ├── config.py                # пути, модели, параметры, настройки
│   ├── features.py               # кастомные трансформеры для фиче-инжиниринга
│   ├── models.py                 # сборка preprocessor + estimator по конфигу
│   └── my_utils.py               # вспомогательные утилиты
│
├── main.py                      # точка входа
├── pyproject.toml
└── README.md
```

### Требования

Python и все зависимости ставятся автоматически через `uv sync` - версии зафиксированы в `pyproject.toml` / `uv.lock`, отдельно ничего устанавливать не нужно.