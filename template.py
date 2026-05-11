import os
from pathlib import Path

list_of_files = [
    "src/__init__.py",
    "src/exception.py",
    "src/logger.py",
    "src/utils.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_monitor.py",
    "src/pipeline/__init__.py",
    "src/pipeline/train_pipeline.py",
    "src/pipeline/predict_pipeline.py",
    "notebook/EDA.ipynb",
    "notebook/model_training.ipynb",
    "templates/index.html",
    "app.py",
    "main.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = filepath.parent

    if filedir != Path("."):
        os.makedirs(filedir, exist_ok=True)

    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            pass