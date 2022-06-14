#!/usr/bin/env python3

import argparse
import ray
from train import train


def hyperopt() -> None:
    """Script's main function; runs hyperoptimization of 3D-CNN for given task."""

    hyperopt_args = vars(get_argsparse().parse_args())

    config = {
        "num_hidden_layers": ray.tune.grid_search([3, 4, 5]),
        "num_filters": ray.tune.grid_search([4, 8, 16]),
        "filter_size": ray.tune.grid_search([3, 5]),
        "batch_size": ray.tune.grid_search([32, 64]),
        "learning_rate": ray.tune.grid_search([1e-4, 1e-3]),
        "dropout": ray.tune.grid_search([0.25, 0.5]),
        "task": hyperopt_args["task"],
        "data_dir": hyperopt_args["data_dir"],
        "num_epochs": 40,
        "num_runs": 1,
        "run": -1,
        "num_folds": 3,
        "fold": -1,
        "log_dir": hyperopt_args["log_dir"],
        "run_group_name": "none",
        "wandb_entity": "athms",
        "wandb_project": "interpretability-comparison",
        "wandb_mode": "offline",
        "report_to": "tune",
        "smoke_test": False,
        "verbose": False,
        "seed": 1,
        "model_config": "none"
    }

    result = ray.tune.run(
        train,
        resources_per_trial={
            "cpu": hyperopt_args["cpus_per_trial"],
            "gpu": hyperopt_args["gpus_per_trial"]
        },
        config=config,
        local_dir="results/hyperopt",
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    return None


def get_argsparse(parser: argparse.ArgumentParser=None) -> argparse.ArgumentParser:
    
    if parser is None:
        parser = argparse.ArgumentParser(
            description='hyperopt CNN for given task'
        )

    parser.add_argument(
        '--task',
        metavar='STR',
        default='WM',
        type=str,
        help='task for which CNN is optimized '
             '(default: WM)'
    )

    parser.add_argument(
        '--data-dir',
        metavar='DIR',
        default='data/task-WM/trial_images',
        type=str,
        help='path tp trial-level BOLD GLM maps'
    )

    parser.add_argument(
        '--log-dir',
        metavar='DIR',
        default='results/hyperopt/task-WM',
        type=str,
        help='path where models and logs are stored'
    )

    parser.add_argument(
        '--cpus-per-trial',
        metavar='N',
        default=2,
        type=int,
        help='number of CPUs per tune hyperopt trial'
    )

    parser.add_argument(
        '--gpus-per-trial',
        metavar='N',
        default=1,
        type=float,
        help='number of GPUs per tune hyperopt trial'
    )

    return parser


if __name__ == '__main__':

    ray.shutdown()
    ray.init()

    hyperopt()