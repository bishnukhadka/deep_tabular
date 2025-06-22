from dataclasses import dataclass, asdict
import json
from dataclasses import dataclass, asdict
from typing import List
import pandas as pd
import openml
import argparse

"""
Fetches datasets and metadata from an OpenML benchmark suite by suite ID.
Returns each dataset as a dataclass containing metadata, features (DataFrame), 
and target (Series) for easy benchmarking and analysis.
"""

@dataclass
class DatasetMetadata:
    """
    Stores metadata information about a dataset.

    Attributes:
        name (str): Name of the dataset.
        task_id (int): OpenML task ID associated with the dataset.
        dataset_id (int): OpenML dataset ID.
        num_features (int): Number of features/columns in the dataset.
        num_samples (int): Number of samples/rows in the dataset.
        categorical_features (List[str]): List of feature names that are categorical.
        target_name (str): Name of the target variable/column.
    """
    name: str
    task_id: int
    dataset_id: int
    num_features: int
    num_samples: int
    categorical_features: List[str]
    target_name: str

    def __str__(self):
        # Class name + pretty printed JSON
        json_str = json.dumps(asdict(self), indent=4)
        return f"{self.__class__.__name__}:\n{json_str}"


@dataclass
class DatasetBundle:
    """
    Combines dataset metadata with the actual data.

    Attributes:
        metadata (DatasetMetadata): Metadata information about the dataset.
        X (pd.DataFrame): Features/data matrix.
        y (pd.Series): Target variable/labels.
    """
    metadata: DatasetMetadata
    X: pd.DataFrame
    y: pd.Series

def fetch_datasets_from_suite(suite_id: int) -> List[DatasetBundle]:
    """
    Fetches all datasets from an OpenML benchmark suite by suite ID.

    This function downloads all tasks associated with the given OpenML
    benchmark suite ID. For each task, it retrieves the dataset,
    extracts features and targets as pandas DataFrame and Series,
    identifies categorical features, and collects dataset metadata.

    Args:
        suite_id (int): OpenML benchmark suite ID.

    Returns:
        List[DatasetBundle]: A list of DatasetBundle objects, each containing
                            metadata and the corresponding dataset (X, y).
    """
    try:
        benchmark_suite = openml.study.get_suite(suite_id)
    except Exception as e:
        print(f"Failed to get suite with ID {suite_id}: {e}")
        return []
    
    dataset_bundles = []

    assert benchmark_suite is not None

    for task_id in benchmark_suite.tasks:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format='dataframe',
            target=task.target_name
        )

        categorical_features = [
            name for name, is_cat in zip(attribute_names, categorical_indicator) if is_cat
        ]

        metadata = DatasetMetadata(
            name=dataset.name,
            task_id=task.task_id,
            dataset_id=dataset.dataset_id,
            num_features=X.shape[1],
            num_samples=X.shape[0],
            categorical_features=categorical_features,
            target_name=task.target_name
        )

        dataset_bundles.append(DatasetBundle(metadata=metadata, X=X, y=y))

    return dataset_bundles

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch datasets from OpenML benchmark suite.')
    parser.add_argument('suite_id', type=int, help='OpenML benchmark suite ID (e.g., 336)')
    args = parser.parse_args()

    bundles = fetch_datasets_from_suite(args.suite_id)

    # Preview the third dataset's metadata and first few rows of data
    bundle = bundles[0]
    print(bundle.metadata)
    print(bundle.X.head())  # Prints the first few rows as plain text



