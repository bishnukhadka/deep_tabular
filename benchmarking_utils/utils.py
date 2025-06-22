import importlib
import numpy as np
import torch
import random
from dataclasses import is_dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
from typing import Any, Dict, Tuple, List, Optional, Union
import optuna

import sys
import os

# Assuming current working directory is /mnt/d/research or you know the path to it
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_folder not in sys.path:
    sys.path.insert(0, parent_folder)

def suggest_from_space(trial, space, prefix=""):
    params = {}

    for key, val in space.__dict__.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if is_dataclass(val):
            # Recurse into nested dataclass
            nested_params = suggest_from_space(trial, val, prefix=full_key)
            params[key] = nested_params
        elif isinstance(val, tuple):
            dist_type = val[0]

            if dist_type == "int":
                _, low, high, step = val
                params[key] = trial.suggest_int(full_key, low, high, step=step)

            elif dist_type == "float":
                _, low, high = val
                params[key] = trial.suggest_float(full_key, low, high)

            elif dist_type == "logfloat":
                _, low, high = val
                params[key] = trial.suggest_float(full_key, low, high, log=True)

            elif dist_type == "discrete_float":
                _, low, high, step = val
                params[key] = trial.suggest_float(full_key, low, high, step=step)

            elif dist_type == "categorical":
                _, choices = val
                params[key] = trial.suggest_categorical(full_key, choices)

            elif dist_type == "bool":
                params[key] = trial.suggest_categorical(full_key, [True, False])

            elif dist_type == "uniform":
                _, low, high = val
                params[key] = trial.suggest_float(full_key, low, high)

            elif dist_type == "loguniform":
                _, low, high = val
                params[key] = trial.suggest_float(full_key, low, high, log=True)

            elif dist_type == "discrete_uniform":
                _, low, high, step = val
                params[key] = trial.suggest_float(full_key, low, high, step=step)

            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")
        else:
            params[key] = val

    return params

def import_and_init_optspace(name: str)->Any:
    """
    Import the OptSpace class along with ModelConfig and TrainingConfig
    from the module corresponding to `name`.
    Instantiate ModelConfig and TrainingConfig with defaults,
    then instantiate and return the OptSpace instance.

    Args:
        name (str): Name of the opt space (e.g. 'mambular', 'mlp')

    Returns:
        instance: Initialized OptSpace instance with configs set.
    """
    name_lower = name.lower()
    module_path = f"benchmarking_utils.opt_space.{name_lower}_optspace"
    class_name = name_lower.capitalize() + "OptSpace"

    # Import module
    module = importlib.import_module(module_path)

    # Get classes
    OptSpaceClass = getattr(module, class_name)
    ModelConfig = getattr(module, "ModelConfig")
    TrainingConfig = getattr(module, "TrainingConfig")

    # Instantiate configs
    model_config = ModelConfig()
    training_config = TrainingConfig()

    # Instantiate and return OptSpace instance
    instance = OptSpaceClass(model=model_config, training=training_config)
    return instance

def import_model_class(module_name: str, task: str)->Any:
    """
    Import a user-facing model class from 'mambular.models.<module_name>' based on the task.
    - Only 'regression' or 'classification' are supported.
    - Internal classes like 'Sklearn*' are skipped.
    """

    task_suffix_map = {
        "regression": "Regressor",
        "classification": "Classifier"
    }

    if task not in task_suffix_map:
        raise ValueError(f"Unsupported task '{task}'. Use one of: {list(task_suffix_map.keys())}")

    suffix = task_suffix_map[task]

    def is_valid_model_class(name, obj):
        return isinstance(obj, type) and name.endswith(suffix) and not name.startswith("Sklearn")

    full_module_path = f"mambular.models.{module_name}"

    try:
        module = importlib.import_module(full_module_path)
    except ModuleNotFoundError:
        raise ImportError(f"Module '{full_module_path}' not found.")

    class_names = [
        name for name, obj in vars(module).items()
        if is_valid_model_class(name, obj)
    ]

    if not class_names:
        raise ValueError(f"No valid user-defined class ending with '{suffix}' found in module '{full_module_path}'.")

    class_name = class_names[0]
    return getattr(module, class_name)


# Adapted from https://github.com/LAMDA-Tabular/TALENT
#  ---- import from lib.util -----------
def set_seeds(base_seed: int, one_cuda_seed: bool = False) -> None:
    """
    Set random seeds for reproducibility.

    :base_seed: int, base seed
    :one_cuda_seed: bool, whether to set one seed for all GPUs
    """
    assert 0 <= base_seed < 2 ** 32 - 10000
    random.seed(base_seed)
    np.random.seed(base_seed + 1)
    torch.manual_seed(base_seed + 2)
    cuda_seed = base_seed + 3
    if one_cuda_seed:
        torch.cuda.manual_seed_all(cuda_seed)
    elif torch.cuda.is_available():
        # the following check should never succeed since torch.manual_seed also calls
        # torch.cuda.manual_seed_all() inside; but let's keep it just in case
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        # Source: https://github.com/pytorch/pytorch/blob/2f68878a055d7f1064dded1afac05bb2cb11548f/torch/cuda/random.py#L109
        for i in range(torch.cuda.device_count()):
            default_generator = torch.cuda.default_generators[i]
            default_generator.manual_seed(cuda_seed + i)

def create_run_dir(base_dir: str = "runs") -> Path:
    """
    Create a timestamped directory inside the base directory for saving run data.

    Args:
        base_dir (str): Base directory to create the run directory in. Defaults to "runs".

    Returns:
        Path: Path object pointing to the newly created run directory.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = Path(base_dir) / f"optuna_run_{ts}"
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path


def save_json(data: Any, filepath: Path) -> None:
    """
    Save a Python object to a JSON file.

    Args:
        data (Any): The data to save (must be JSON serializable).
        filepath (Path): Path to the file where JSON data will be saved.
    """
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def log_trial(log_file: Path, trial_number: int, params: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    Log a single trial's parameters and results to a JSON log file.

    Args:
        log_file (Path): Path to the JSON log file.
        trial_number (int): Trial number.
        params (Dict[str, Any]): Parameters used in this trial.
        result (Dict[str, Any]): Result metrics from the trial.
    """
    log_entry = {
        "trial_number": trial_number,
        "parameters": params,
        "result": result
    }

    if log_file.exists():
        with open(log_file, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)
    save_json(logs, log_file)

def objective_factory(
    model_name: str, 
    task: str, 
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    log_file: Path, 
    max_epochs : int=5, 
) -> Any:
    """
    Create an Optuna objective function that trains and evaluates the model,
    then logs the trial information.

    Args:
        X_train (Any): Training input data.
        y_train (Any): Training target data.
        X_val (Any): Validating input data.
        y_val (Any): Testing target data.
        log_file (Path): Path to the log file to save trial info.

    Returns:
        Callable: The objective function to be passed to Optuna's study.optimize().
    """


    def objective(trial: optuna.Trial) -> float:

        task_suffix_map = {
        "regression": "Regressor",
        "classification": "Classifier"
        }

        task_lower = task.lower()
        assert task_lower in task_suffix_map, (
            f"Invalid task '{task}'. Only 'regression' and 'classification' are supported."
        )

        model_lower = model_name.lower()
        
        space = import_and_init_optspace(model_lower)
        params = suggest_from_space(trial, space)

        model_class = import_model_class(model_name, task)
        model = model_class(
            **params['model']
        )

        model.fit(
            X_train[:10], y_train[:10],
            max_epochs=max_epochs,
            **params['training']
        )

        result = model.evaluate(X_val[:10], y_val[:10])
        log_trial(log_file, trial.number, params, result)
        if task=="regression":
            return result['Mean Squared Error']
        else: 
            return result['Accuracy']
    return objective


def tune_hyper_parameters(
    model_name: str, 
    task: str, 
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    n_trials: int = 5,
    base_dir: str = "runs"
) -> Tuple[optuna.Study, Path]:
    """
    Run an Optuna hyperparameter optimization study on the model.

    Args:
        X_train (Any): Training input data.
        y_train (Any): Training target data.
        X_val (Any): Validating input data.
        y_val (Any): Validating target data.
        n_trials (int): Number of Optuna trials. Defaults to 50.
        base_dir (str): Base directory to save runs and logs. Defaults to "runs".

    Returns:
        Tuple[optuna.Study, Path]: The Optuna study object and the directory path where logs are saved.
    """
    run_dir = create_run_dir(base_dir)
    log_file = run_dir / "optuna_trials_log.json"
    best_params_file = run_dir / "best_params.json"
    best_trial_file = run_dir / "best_trial.json"

    objective = objective_factory(
        model_name, 
        task, 
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        log_file,
        max_epochs=5
        )

    direction = "minimize" if task == "regression" else "maximize"  # or metric-based
    study = optuna.create_study(direction=direction)

    study.optimize(
            objective, 
            n_trials=n_trials,
            show_progress_bar=True)

    # Save best results
    save_json(study.best_params, best_params_file)
    best_trial_data = {
        "trial_number": study.best_trial.number,
        "value": study.best_trial.value,
        "params": study.best_trial.params,
    }
    save_json(best_trial_data, best_trial_file)

    print('Best Hyper-Parameters')
    print(best_trial_data)

    return study, run_dir


# inspired by LAMDA-TALENT
def summarize_results(
    results_list: List[Dict[str, Any]], 
    loss_name: str, 
    model_name: Optional[str] = "mambular",
    path: Optional[Union[str, Path]] = None
) -> None:
    
    # Prepare a list to capture output lines
    output_lines = []

    def print_and_capture(text: str = ""):
        print(text)
        output_lines.append(text)

    all_metric_names = set()
    time_key = "training_time"
    for res in results_list:
        all_metric_names.update([k for k in res.keys() if k != time_key])
    all_metric_names = sorted(all_metric_names)

    times = []
    missing_time = False
    for res in results_list:
        if time_key in res:
            times.append(res[time_key])
        else:
            missing_time = True
            times.append(np.nan)
    if missing_time:
        print_and_capture(f"Warning: Some runs are missing '{time_key}' key, NaN will be used for time.")

    metric_arrays = {name: [] for name in all_metric_names}
    for res in results_list:
        for name in all_metric_names:
            metric_arrays[name].append(res.get(name, np.nan))
    metric_arrays['Time'] = times
    all_metric_names.append('Time')

    mean_metrics = {name: np.nanmean(metric_arrays[name]) for name in all_metric_names}
    std_metrics = {name: np.nanstd(metric_arrays[name]) for name in all_metric_names}

    loss_values = metric_arrays.get(loss_name, [])
    mean_loss = np.nanmean(loss_values) if loss_values else float('nan')
    std_loss = np.nanstd(loss_values) if loss_values else float('nan')

    print_and_capture(f"\n{model_name}: {len(results_list)} Trials Summary")
    print_and_capture("-" * 40)

    for name in all_metric_names:
        vals = metric_arrays[name]
        if name == loss_name:
            formatted_results = ', '.join([f"{v:.8e}" if not np.isnan(v) else "nan" for v in vals])
            print_and_capture(f"Loss ({name}) values: {formatted_results}")
            print_and_capture(f"Loss ({name}) MEAN = {mean_loss:.8e} ± {std_loss:.8e}\n")
        elif name == 'Time':
            formatted_results = ', '.join([f"{v:.4f}" if not np.isnan(v) else "nan" for v in vals])
            print_and_capture(f"Time values (s): {formatted_results}")
            print_and_capture(f"Mean Time = {mean_metrics[name]:.4f} ± {std_metrics[name]:.4f}\n")
        else:
            formatted_results = ', '.join([f"{v:.6f}" if not np.isnan(v) else "nan" for v in vals])
            print_and_capture(f"{name} Results: {formatted_results}")
            print_and_capture(f"{name} MEAN = {mean_metrics[name]:.6f} ± {std_metrics[name]:.6f}\n")

    print_and_capture("-" * 20 + " GPU Info " + "-" * 20)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print_and_capture(f"{num_gpus} GPU(s) Available.")
        for i in range(num_gpus):
            gpu_info = torch.cuda.get_device_properties(i)
            print_and_capture(f"GPU {i}: {gpu_info.name}")
            print_and_capture(f"  Total Memory:          {gpu_info.total_memory / 1024**2:.2f} MB")
            print_and_capture(f"  Multi Processor Count: {gpu_info.multi_processor_count}")
            print_and_capture(f"  Compute Capability:    {gpu_info.major}.{gpu_info.minor}")
    else:
        print_and_capture("CUDA is unavailable.")
    print_and_capture("-" * 50)

    # Save to file if path is provided
    if path is not None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        file_path = path / "results.txt"         # Always write to 'results.txt'
        
        with file_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(output_lines))


def get_task_type(suite_id: str) -> str:
    """
    Returns the task type ('regression' or 'classification') based on the suite_id.

    Parameters:
    - suite_id (str): The suite ID to check.

    Returns:
    - str: 'regression' or 'classification'

    Raises:
    - ValueError: If suite_id is not among the known suite IDs.
    """

    regression_suites = {'336', '335'}
    classification_suites = {'337', '334'}

    if suite_id in regression_suites:
        return 'regression'
    elif suite_id in classification_suites:
        return 'classification'
    else:
        raise ValueError(f"Unknown suite_id '{suite_id}'. Must be one of: "
                         f"{', '.join(regression_suites | classification_suites)}")


from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
    max_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
)

REGRESSION_METRICS = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "mape": mean_absolute_percentage_error,    # Percent error, useful for interpretability
    "medae": median_absolute_error,
    "r2": r2_score,
    "evs": explained_variance_score,           # Variance explained by the model
    "max_error": max_error,
}

CLASSIFICATION_METRICS = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "roc_auc": roc_auc_score,
    "log_loss": log_loss,
    "confusion_matrix": confusion_matrix,    # returns matrix instead of scalar, but can be useful
}

def build_eval_metrics(task_type, metric_names):
    if task_type == "regression":
        allowed = REGRESSION_METRICS
    elif task_type == "classification":
        allowed = CLASSIFICATION_METRICS
    else:
        raise ValueError("Unsupported task type. Use 'regression' or 'classification'.")

    eval_metrics = {}
    for name in metric_names:
        if name not in allowed:
            raise ValueError(f"Metric '{name}' is not valid for {task_type} tasks.")
        eval_metrics[name] = allowed[name]

    return eval_metrics


from pathlib import Path
import re

def create_model_exp_dir(base_dir: str, model_name: str, task_type: str) -> Path:
    base_path = Path(base_dir) / model_name / task_type
    base_path.mkdir(parents=True, exist_ok=True)

    existing_exps = [p.name for p in base_path.iterdir() if p.is_dir() and re.match(r"exp(\d*)$", p.name)]

    exp_nums = []
    for name in existing_exps:
        if name == "exp":
            exp_nums.append(0)
        else:
            num = int(name[3:])
            exp_nums.append(num)

    exp_nums_set = set(exp_nums)
    next_num = 0
    while next_num in exp_nums_set:
        next_num += 1

    exp_name = "exp" if next_num == 0 else f"exp{next_num}"
    exp_path = base_path / exp_name
    exp_path.mkdir()
    return exp_path

import pandas as pd
def encode_series(series: pd.Series) -> pd.Series:
    """
    Encode a pandas Series into integer codes representing categories.

    - If the Series is an ordered categorical dtype, returns its existing category codes.
    - Otherwise, encodes unique values in the order they first appear.
    - Converts non-categorical or unordered categorical data to strings before encoding to 
      ensure consistent ordering and handling of mixed types.
    - Preserves the original Series' name and index in the returned Series.

    Parameters:
    -----------
    series : pd.Series
        The input pandas Series to encode.

    Returns:
    --------
    pd.Series
        A Series of integer codes corresponding to categories or unique values.
    """
    if isinstance(series.dtype, pd.CategoricalDtype) and series.cat.ordered:
        return series.cat.codes
    else:
        unique_cats = pd.unique(series.astype(str).values)
        cat = pd.Categorical(series.astype(str).values, categories=unique_cats, ordered=True)
        return pd.Series(cat.codes, name=series.name, index=series.index)