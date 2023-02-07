#!/usr/bin/env python3

import os
import argparse
import json
import shutil
import numpy as np
import pandas as pd


def identify_best_model_configuration(config=None) -> None:
    """Script's main function; identifies best-performing model 
    configuration for given task from hyperopt results."""
    
    if config is None:
        config =  vars(get_argparse().parse_args())

    model_dirs = [
        os.path.join(config['hyperopt_dir'], d)
        for d in os.listdir(config['hyperopt_dir'])
        if d.startswith(f'task-{config["task"]}')
    ]
    model_errors = []

    for i, model_dir in enumerate(model_dirs):
        model_config = json.load(open(os.path.join(model_dir, 'config.json')))
        run_dirs = [
            os.path.join(model_dir, d)
            for d in os.listdir(model_dir)
            if d.startswith('run-')
        ]
        run_train_errors = []
        run_validation_errors = []

        for run_dir in run_dirs:
            fold_dirs = [
                os.path.join(run_dir, d)
                for d in os.listdir(run_dir)
                if d.startswith('fold-')
            ]
            fold_train_errors = []
            fold_validation_errors = []

            for fold_dir in fold_dirs:
                model_train_performance_path = os.path.join(fold_dir, 'train_history.csv')
                model_validation_performance_path = os.path.join(fold_dir, 'validation_history.csv')

                if os.path.exists(model_train_performance_path) and os.path.isfile(model_validation_performance_path):
                    model_train_performance = pd.read_csv(model_train_performance_path)
                    model_validation_performance = pd.read_csv(model_validation_performance_path)
                    best_epoch = model_validation_performance['epoch'].values[np.argmin(model_validation_performance['loss'])]
                    fold_train_errors.append(100 - (model_train_performance[model_train_performance['epoch']==best_epoch, 'accuracy']*100) )
                    fold_validation_errors.append(100 - (model_validation_performance[model_validation_performance['epoch']==best_epoch, 'accuracy']*100) )

            run_train_errors.append(np.mean(fold_train_errors))
            run_validation_errors.append(np.mean(fold_validation_errors))

        model_errors.append(
            pd.DataFrame(
                {
                    'L': int(model_config['num_hidden_layers']),
                    'F': int(model_config['num_filters']),
                    'FS': int(model_config['filter_size']),
                    'BS': int(model_config['batch_size']),
                    'LR': float(model_config['learning_rate']),
                    'D': float(model_config['dropout']),
                    'mean_validation_error': np.mean(run_validation_errors),
                    'mean_train_error':  np.mean(run_train_errors),
                    'difference_mean_error': np.abs(np.mean(run_validation_errors) - np.mean(run_train_errors))
                },
                index=[i]
            )
        )

    model_errors = pd.concat(model_errors)
    model_errors['error'] = model_errors['mean_validation_error'] + model_errors['difference_mean_error']
    model_errors.to_csv(
        os.path.join(
            config['hyperopt_dir'],
            'model-errors.csv'
        ),
        index=False
    )
    model_errors = model_errors.sort_values(by='error')
    best_model = model_errors.iloc[0]
    best_model_dir = (
        f'task-{config["task"]}'
        f'_l-{int(best_model["L"])}'
        f'_f-{int(best_model["F"])}'
        f'_fs-{int(best_model["FS"])}'
        f'_bs-{int(best_model["BS"])}'
        f'_lr-{str(best_model["LR"]).replace(".", "")}'
        f'_d-{str(best_model["D"]).replace(".", "")}'
    )
    best_model_config = os.path.join(
        config['hyperopt_dir'],
        best_model_dir,
        'config.json'
    )
    print('\nBest model:')
    print(f'\tMean validation error: {best_model["mean_validation_error"]}%')
    print(f'\tMean train error: {best_model["mean_train_error"]}%')
    print('\tConfig:')
    for k, v in json.load(open(best_model_config)).items():
        print(f'\t\t {k}: {v}')

    print(f'\nCopying best model config to: {config["hyperopt_dir"]}..')
    shutil.copyfile(best_model_config, os.path.join(config["hyperopt_dir"], 'best_model_config.json'))
    print('Done!')

    return None


def get_argparse(parser: argparse.ArgumentParser=None) -> argparse.ArgumentParser:
    
    if parser is None:
        parser = argparse.ArgumentParser(
            description='identify best-performing model configuration for given task',
        )

    parser.add_argument(
        '--task',
        metavar='TASK',
        default='WM',
        type=str,
        help='name of task (default: WM)'
    )
    parser.add_argument(
        '--hyperopt-dir',
        metavar='DIR',
        default='results/hyperopt/task-WM',
        type=str,
        help='path to hyperopt results (as generated by running scripts/hyperopt.py)'
             '(default: results/hyperopt/task-WM)'
    )
    
    return parser


if __name__ == '__main__':

    identify_best_model_configuration()