from tqdm import tqdm
import argparse
from utils import (
    summarize_results,
    tune_hyper_parameters,
    set_seeds,
    get_task_type,
    import_model_class, 
    build_eval_metrics,
    create_model_exp_dir,
    encode_series,
    REGRESSION_METRICS,
    CLASSIFICATION_METRICS
)
from openml_dataset_loader import fetch_datasets_from_suite
from data_split import get_splits
import json
import os
import time
from datetime import timedelta

script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, "datasets.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking Configuration")

    parser.add_argument('--model_name', type=str, default='mambular',
                        help='Name of the model to use')
    parser.add_argument('--seed_num', type=int, default=4, 
                        help='Random seed for reproducibility')
    parser.add_argument('--suite_id', type=str, required=True,
                        help='ID of the benchmark/task suite')
    parser.add_argument('--tune', action='store_true', default=False,
                        help='Enable hyperparameter tuning')
    parser.add_argument('--n_trials', type=int, default=1,
                        help='Number of trials for tuning or evaluation')
    parser.add_argument('--save_dir', type=str, default="runs",
                        help='Directory where results will be saved')
    parser.add_argument('--max_epochs', type=int, default=1,
                        help='The maximum number of epochs a model is going to be trained for.')
    parser.add_argument(
        "--metrics",
        nargs="+", 
        required=True,
        help="List of metrics to use (space-separated)"
    )
    # will add cross validation in the future if required. 

    return parser.parse_args()

if __name__ == '__main__':
    loss_list, results_list, time_list = [], [], []

    # get args
    args = parse_args()
    print("Arguments received:")
    print(args)
    print(args.metrics)

    task = get_task_type(suite_id=args.suite_id)

    # Get task-specific allowed metrics
    if task == "regression":
        allowed_metrics = REGRESSION_METRICS
        loss_metric = 'mse'
    elif task == "classification":
        allowed_metrics = CLASSIFICATION_METRICS
        loss_metric = 'log_loss'
    else:
        raise ValueError(f"Unsupported task type: {task}")

    result_path=create_model_exp_dir(
        base_dir=args.save_dir,
        model_name=args.model_name,
        task_type = task)

    # Combine all valid metric names
    all_valid_metrics = set(REGRESSION_METRICS.keys()) | set(CLASSIFICATION_METRICS.keys())

    # Validate that all input metrics are known
    invalid_metrics = [m for m in args.metrics if m not in all_valid_metrics]
    assert not invalid_metrics, f"Invalid metric(s): {', '.join(invalid_metrics)}\nValid metric(s): {', '.join(all_valid_metrics)}"

    with open(json_path, "r") as f:
        suite_datasets = json.load(f)

    # get the dataset and get the train, val and test sets. 
    dataset_bundles = fetch_datasets_from_suite(args.suite_id)

    # One unified check — this catches typos and wrong-task metrics
    invalid_metrics = [m for m in args.metrics if m not in allowed_metrics]
    assert not invalid_metrics, (
        f"Invalid metric(s): {', '.join(invalid_metrics)}\n"
        f"Valid metrics for {task}: {', '.join(sorted(allowed_metrics.keys()))}"
    )

    for dataset in dataset_bundles:
        dataset_names = suite_datasets.get(str(args.suite_id), [])
        print(dataset_names)
        print(f'Woking on: {dataset.metadata.name} dataset. {dataset.metadata}. \nTarget type: {type(dataset.y[0])}')

        if dataset.metadata.name not in dataset_names:
            print(f"{dataset.metadata.name} not in suite {args.suite_id}. Skipping or exiting.")
            continue
        
        if task=="classification":
            y = encode_series(dataset.y)
        else:
            y = dataset.y    

        # split without cross-validation
        ((X_train, y_train, X_val, y_val), X_test, y_test) = get_splits(
            dataset.X, 
            y, 
            cv = False,
            test_size = 0.1,
            val_size = 0.1,
            random_state = args.seed_num,
        )
        # print(f'{x}:{dataset.metadata.name}')
        print(f'{dataset.metadata.name}')

        if args.tune:
            (study, saved_path) = tune_hyper_parameters(
                model_name=args.model_name, 
                task=task, 
                X_train=X_train, 
                y_train=y_train, 
                X_val=X_val, 
                y_val=y_val, 
                n_trials=args.n_trials, 
                base_dir= str(result_path)
            )
            print(study.best_params)
        
        ## Training Stage over different random seeds
        for seed in tqdm(range(args.seed_num)):
            args.seed = seed    # update seed  
            set_seeds(args.seed)

            # get the model based on the task and the model_name
            model = import_model_class(args.model_name, task)()
            print(f"Model instance: {model}")            
            # get the training time.
            # time_start 
            start_time = time.time()

            model.fit(
                X_train, 
                y_train, 
                max_epochs = args.max_epochs
            )    

            # time_stop
            end_time = time.time()
            time_cost = end_time - start_time

            # elapsed = timedelta(seconds=time_cost)
            # print(f'\n\n\n Time Elapsed: {elapsed}')
            
            # add loss metric 
            metrics = args.metrics
            metrics_set = set(metrics)
            if loss_metric not in metrics_set:
                metrics.append(loss_metric)

            eval_metrics = build_eval_metrics(task, args.metrics)
            if task=="classification":
                eval_metrics = {k: (v, False) for k, v in eval_metrics.items()}

            result = model.evaluate(X_test, y_test, metrics=eval_metrics)
            result['training_time'] = time_cost
            results_list.append(result)

    summarize_results(results_list=results_list,
                      loss_name=loss_metric,
                      model_name=args.model_name,
                      path=result_path
                      )