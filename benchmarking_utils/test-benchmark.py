import argparse
import os
import random
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Configuration")

    parser.add_argument('--model_name', type=str, default='mambular',
                        help='Name of the model to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--suite_id', type=str, required=True,
                        help='ID of the benchmark/task suite')
    parser.add_argument('--tune', action='store_true', default=False, 
                        help='Enable hyperparameter tuning')
    parser.add_argument('--n_trials', type=int, default=10,
                        help='Number of trials for tuning or evaluation')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory where results will be saved')

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)  # Uncomment if using PyTorch
    print(f"Seed set to: {seed}")

def main():
    args = parse_args()

    print("Arguments received:")
    print(args)

    # Create save directory if it doesn't exist
    # os.makedirs(args.save_dir, exist_ok=True)
    print(f"Results will be saved to: {args.save_dir}")

    # Set random seed
    set_seed(args.seed)

    # Place your main logic here
    if args.tune:
        print(f"Tuning enabled. Running {args.n_trials} trials...")
        # Your tuning logic here
    else:
        print(f"Running evaluation with model {args.model_name} on suite {args.suite_id}")
        # Your evaluation logic here

if __name__ == '__main__':
    main()
