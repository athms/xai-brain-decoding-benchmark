#!/usr/bin/env python3

import os
import argparse
import ray
import train


def hyperopt() -> None:
    """Script's main function; runs 3D-CNN hyperoptimization with tune for given task."""

    hyperopt_config = vars(get_argparse().parse_args())
    train_config = train.make_config()

    config = {
        "num_hidden_layers": ray.tune.grid_search([3, 4, 5]),
        "num_filters": ray.tune.grid_search([4, 8, 16]),
        "filter_size": ray.tune.grid_search([3, 5]),
        "batch_size": ray.tune.grid_search([32, 64]),
        "learning_rate": ray.tune.grid_search([1e-4, 1e-3]),
        "dropout": ray.tune.grid_search([0.25, 0.5]),
        "task": hyperopt_config["task"],
        "data_dir": hyperopt_config["data_dir"],
        "num_epochs": 40,
        "num_runs": 1,
        "num_folds": 3,
        "log_dir": hyperopt_config["log_dir"],
        "report_to": "tune",
        "seed": 1234
    }

    for k, v in train_config.items():
        
        if k not in config:
            config[k] = v

    _ = ray.tune.run(
        train.train,
        resources_per_trial={
            "cpu": hyperopt_config["cpus_per_trial"],
            "gpu": hyperopt_config["gpus_per_trial"]
        },
        config=config,
        local_dir=os.path.join(
            hyperopt_config["log_dir"],
            'ray_results'
        ),
    )

    return None


def get_argparse(parser: argparse.ArgumentParser=None) -> argparse.ArgumentParser:
    
    if parser is None:
        parser = argparse.ArgumentParser(
            description='hyperopt 3D-CNN for given task'
        )

    parser.add_argument(
        '--task',
        metavar='STR',
        default='WM',
        type=str,
        help='task for which 3D-CNN is optimized '
             '(default: WM)'
    )
    parser.add_argument(
        '--data-dir',
        metavar='DIR',
        default='data/task-WM/trial_images',
        type=str,
        help='path to trial-level BOLD GLM maps'
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
        default=4,
        type=int,
        help='number of CPUs per hyperopt trial'
    )
    parser.add_argument(
        '--gpus-per-trial',
        metavar='N',
        default=1,
        type=float,
        help='number of GPUs per hyperopt trial'
    )

    return parser


if __name__ == '__main__':

    ray.shutdown()
    ray.init()

    hyperopt()