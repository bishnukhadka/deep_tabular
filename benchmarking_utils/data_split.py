from typing import List, Tuple, Optional, Union
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

def get_splits(
    X: pd.DataFrame,
    y: pd.Series,
    cv: bool = False,
    n_splits: int = 5,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: Optional[pd.Series] = None,
) -> Union[
    Tuple[Tuple, pd.DataFrame, pd.Series],
    Tuple[List[Tuple], pd.DataFrame, pd.Series]
]:
    """     
    Splits data for training/validation/testing.

    Args:
        X: Features dataframe
        y: Target series
        cv: Whether to perform cross-validation
        n_splits: Number of CV folds (if cv=True)
        test_size: Fraction of data as test set
        val_size: Fraction of remaining data as validation set (used only if cv=False)
        random_state: Random seed for reproducibility
        stratify: Optional stratification target (default: y)

    Returns:
        - If cv=False: Tuple ((X_train, y_train, X_val, y_val), X_test, y_test)
        - If cv=True: Tuple (list of CV folds with train/val splits, X_test, y_test)
    """
    stratify = stratify if stratify is not None else y

    def can_stratify(y_):
        return all(count >= 2 for count in y_.value_counts())

    stratify_param = stratify if can_stratify(stratify) else None

    if not cv:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify_param, random_state=random_state
        )
        val_adjusted = val_size / (1 - test_size)
        stratify_param_train = y_temp if can_stratify(y_temp) else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_adjusted, stratify=stratify_param_train, random_state=random_state
        )
        return ((X_train, y_train, X_val, y_val), X_test, y_test)

    else:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify_param, random_state=random_state
        )
        cross_validation = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
        # For CV stratified k-fold, you might want to skip stratification if not possible
        if not can_stratify(y_trainval):
            raise ValueError("Cannot perform StratifiedKFold because of classes with fewer than 2 samples.")

        splits = []
        for train_idx, val_idx in cross_validation.split(X_trainval, y_trainval):
            X_train_fold = X_trainval.iloc[train_idx]
            y_train_fold = y_trainval.iloc[train_idx]
            X_val_fold = X_trainval.iloc[val_idx]
            y_val_fold = y_trainval.iloc[val_idx]
            splits.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
        return (splits, X_test, y_test)