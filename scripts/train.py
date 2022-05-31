#!/usr/bin/env python3

import os, sys
import argparse
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import json
import torch
import wandb
from ray import tune
sys.path.append('./')
import src


def get_argparse(parser: argparse.ArgumentParser=None) -> argparse.ArgumentParser:
    
    if parser is None:
        parser = argparse.ArgumentParser(
            description='train CNN for given task'
        )

    parser.add_argument(
        '--task',
        metavar='TASK',
        default='WM',
        type=str,
        help='task for which CNN is trained '
             '(default: WM)'
    )
    parser.add_argument(
        '--data-dir',
        metavar='DIR',
        default='data/task-WM/trial_images',
        type=str,
        required=False,
        help='path where subject images are stored '
             '(default: data/)'
    )


    parser.add_argument(
        '--num-hidden-layers',
        metavar='N',
        default=4,
        type=int,
        required=False,
        help='number of hidden CNN layers '
             '(default: 5)'
    )
    parser.add_argument(
        '--num-filters',
        metavar='N',
        default=8,
        type=int,
        required=False,
        help='number of CNN kernels per layer '
             '(default: 8)'
    )
    parser.add_argument(
        '--filter-size',
        metavar='N',
        default=3,
        type=int,
        required=False,
        help='size of 3D-CNN filter per channel '
             '(default: 3)'
    )


    parser.add_argument(
        '--batch-size',
        metavar='N',
        default=32,
        type=int,
        required=False,
        help='number of samples per training step '
             '(default: 32)'
    )
    parser.add_argument(
        '--num-epochs',
        metavar='N',
        default=40,
        type=int,
        required=False,
        help='number of training epochs '
             '(default: 10)'
    )
    parser.add_argument(
      '--learning-rate',
      metavar='LR',
      default=1e-4,
      type=float,
      required=False,
      help='learning rate for AdamW '
           '(default: 1e-4)'
    )
    parser.add_argument(
      '--dropout',
      metavar='N',
      default=0.25,
      type=float,
      required=False,
      help='Dropout rate '
           '(default: 0.25)'
    )
    parser.add_argument(
      '--num-runs',
      metavar='N',
      default=10,
      type=int,
      required=False,
      help='Number of training runs '
           '(default: 10)'
    )
    parser.add_argument(
      '--num-folds',
      metavar='N',
      default=3,
      type=int,
      required=False,
      help='Number of CV-folds '
           '(default: 3)'
    )
    parser.add_argument(
      '--run',
      metavar='N',
      default=-1,
      type=int,
      required=False,
      help='run to compute (overwrites num-runs) '
           '(default: -1)'
    )
    parser.add_argument(
      '--fold',
      metavar='N',
      default=-1,
      type=int,
      required=False,
      help='cv-fold to compute (overwrites num-folds) '
           '(default: -1)'
    )


    parser.add_argument(
        '--log-dir',
        metavar='DIR',
        default='results/models/',
        type=str,
        required=False,
        help='path where outputs are stored '
             '(default: results/models/)'
    )
    parser.add_argument(
        '--run-group-name',
        metavar='NAME',
        default='none',
        type=str,
        required=False,
        help='group name for runs '
             '(default: none)'
    )
    parser.add_argument(
        '--report-to',
        metavar='REPORT_TP',
        default='none',
        type=str,
        choices=('none', 'wandb', 'tune'),
        help='whether to report training results to wandb, tune or none '
             '(default: none)'
    )
    parser.add_argument(
        '--wandb-entity',
        metavar='WANDB_ENTITY',
        default='athms',
        type=str,
        help='wandb entity '
             '(default: athms)'
    )
    parser.add_argument(
        '--wandb-project',
        metavar='WANDB_PROJ',
        default='interpretability-comparison',
        type=str,
        help='wandb project '
             '(default: interpretability-comparison)'
    )
    parser.add_argument(
        '--wandb-mode',
        metavar='WANDB_MODE',
        default='online',
        choices=('online', 'offline', 'disabled'),
        type=str,
        help='wandb mode '
             '(default: online)'
    )
    parser.add_argument(
        '--smoke-test',
        metavar='TEST',
        default='False',
        choices=('True', 'False'),
        type=str,
        help='whether to run training as smoke-test '
             '(default: False)'
    )
    parser.add_argument(
        '--verbose',
        metavar='VERBOSE',
        default='True',
        choices=('True', 'False'),
        type=str,
        help='comment training? '
             '(default: True)'
    )
    parser.add_argument(
        '--seed',
        metavar='DIR',
        default=1,
        type=int,
        required=False,
        help='initial random seed '
             '(default: 1)'
    )
    parser.add_argument(
        '--permute-labels',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        required=False,
        help='whether or not to permute training labels'
    )
    parser.add_argument(
        '--model-config',
        metavar='CONFIG',
        default='none',
        type=str,
        required=False,
        help=''
    )

    return parser


def config_cleanup(config):
    config["verbose"] = config["verbose"] == 'True'
    config["smoke_test"] = config["smoke_test"] == 'True'
    config["permute_labels"] = config["permute_labels"] == 'True'
    config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    if config["smoke_test"]:
        config["num_epochs"] = 1
        config["batch_size"] = 2
        config["num_runs"] = 1
        config["num_folds"] = 2
    
    if config["fold"] != -1:
        config["num_folds"] = 1

    if config["run"] != -1:
        config["num_runs"] = 1

    if config['model_config'] != 'none':
        print(f'\n! Loading model config from file: {config["model_config"]}...')
        assert os.path.isfile(config['model_config']), f'{config["model_config"]} not found'
        loaded_config = json.load(open(config['model_config']))
        for key in [
            'num_hidden_layers',
            'num_filters', 
            'filter_size',
            'batch_size',
            'learning_rate',
            'dropout'
        ]:
            print(f'\tsetting {key} to {loaded_config[key]}')
            config[key] = loaded_config[key]

    if config["run_group_name"] == 'none':
        config["run_group_name"] = f'task-{config["task"]}'
        config["run_group_name"] += f'_l-{config["num_hidden_layers"]}'
        config["run_group_name"] += f'_f-{config["num_filters"]}'
        config["run_group_name"] += f'_fs-{config["filter_size"]}'
        config["run_group_name"] += f'_bs-{config["batch_size"]}'
        config["run_group_name"] += f'_lr-{str(config["learning_rate"]).replace(".", "")}'
        config["run_group_name"] += f'_d-{str(config["dropout"]).replace(".", "")}'

    config["log_dir"] = os.path.join(
        config["log_dir"],
        config["run_group_name"]
    )

    return config


def train(config: Dict=None) -> None:

    if config is None:
        config =  vars(get_argparse().parse_args())
    config = config_cleanup(dict(config))
    
    assert config["task"] in [
        'heat-rejection',
        'WM',
        'MOTOR'
    ], f'{config["task"]} is not a valid task'
    
    assert config["report_to"] in [
        'none',
        'wandb',
        'tune'
    ], f'{config["report_to"]} is not a "report_to" value'
    
    os.makedirs(
        config["log_dir"],
        exist_ok=True
    )
    config_filepath = os.path.join(
        config["log_dir"],
        'config.json'
    )
    with open(config_filepath, 'w') as f:
        json.dump(config, f, indent=2)

    train_images, train_labels = load_train_data(config=config)

    train_history = []
    validation_history = []

    in_fold = int(config["fold"])
    in_run = int(config["run"])
    in_run_group_name = str(config["run_group_name"])

    run_iterator = range(config["num_runs"]) if in_run == -1 else [in_run]
    final_eval_losses = []
    final_eval_accuracies = []

    for run in run_iterator:
        config["run"] = run
        config["run_group_name"] = f'{in_run_group_name}_run-{run}' if config["num_folds"] > 1 else in_run_group_name
        
        if config["verbose"]:
            print(
                f'\nTraining run: {run} ...'
            )
        
        fold_iterator = range(config["num_folds"]) if in_fold == -1 else [in_fold]

        for fold in fold_iterator:
            config["fold"] = int(fold)
            
            if config["verbose"]:
                print(
                    f'\tFold: {fold} ...'
                )
            
            dir_addon = f'fold-{config["fold"]}' if config["num_folds"] > 1 else ''
            dir_addon = f"{dir_addon}smoke_test" if config["smoke_test"] else dir_addon
            config["run_log_dir"] = os.path.join(
                config["log_dir"],
                f'run-{run}',
                dir_addon
            )
            os.makedirs(
                config["run_log_dir"],
                exist_ok=True
            )
            if config["report_to"] == 'wandb':
                src.reset_wandb_env()
                wandb_run = wandb.init(
                    config=config,
                    project=str(config["wandb_project"]),
                    entity=str(config["wandb_entity"]),
                    group=str(config["run_group_name"]),
                    name=f'fold-{config["fold"]}'
                        if config["num_folds"] > 1
                        else f'run-{config["run"]}',
                    dir=str(config["run_log_dir"]),
                    job_type='CV'
                        if config["num_folds"] > 1
                        else 'train',
                    mode=str(config["wandb_mode"])
                )

            run_train_history, run_validation_history = train_run(
                config=config,
                images=train_images,
                labels=train_labels,
                wandb_run=wandb_run if config["report_to"] == 'wandb' else None,
            )
            if config["report_to"] == 'tune':
                final_eval_losses.append(run_validation_history['loss'].values[-1])
                final_eval_accuracies.append(run_validation_history['accuracy'].values[-1])

            run_train_history.to_csv(
                os.path.join(
                    config["run_log_dir"],
                    'train_history.csv'
                ),
                index=False
            )
            train_history.append(run_train_history)
            pd.concat(train_history, ignore_index=True).to_csv(
                os.path.join(
                    config["log_dir"],
                    'train_history.csv'
                ),
                index=False
            )
            run_validation_history.to_csv(
                os.path.join(
                    config["run_log_dir"],
                    'validation_history.csv'
                ),
                index=False
            )
            validation_history.append(run_validation_history)
            pd.concat(validation_history, ignore_index=True).to_csv(
                os.path.join(
                    config["log_dir"],
                    'validation_history.csv'
                ),
                index=False
            )
            
            if config["report_to"] == 'wandb':
                wandb_run.finish()

    if config["report_to"] == 'tune':
        tune.report(
            loss=np.mean(final_eval_losses),
            accuracy=np.mean(final_eval_accuracies),
        )

    return None


def load_train_data(config):

    trial_image_paths_split_path = os.path.join(
        config["log_dir"],
        'trial_image_paths.json'
    )

    if not os.path.isfile(trial_image_paths_split_path):
        subjects = np.unique(
            [
                s.split('sub_')[1]
                for s in os.listdir(config["data_dir"])
                if s.startswith('sub_')
            ]
        )
        subjects.sort()
        test_subjects = subjects[::5]
        train_subjects = np.array(
            [
                s for s in subjects
                if s not in test_subjects
            ]
        )
        train_image_paths = src.data.get_subject_trial_image_paths(
            path=config["data_dir"],
            subjects=train_subjects,
            decoding_targets=src.target_labeling[config["task"]].keys()
        )
        test_image_paths = src.data.get_subject_trial_image_paths(
            path=config["data_dir"],
            subjects=test_subjects,
            decoding_targets=src.target_labeling[config["task"]].keys()
        )
        trial_image_paths_split = {
            'train': train_image_paths,
            'test': test_image_paths
        }

        with open(trial_image_paths_split_path, 'w') as f:
            json.dump(trial_image_paths_split, f, indent=2)

    else:

        with open(trial_image_paths_split_path, 'r') as f:
            trial_image_paths_split = json.load(f)

    train_images, train_labels = src.data.load_data(
        image_paths=trial_image_paths_split['train'],
        return_fdata=True,
        target_labeling=src.target_labeling[config["task"]]
    )
    assert train_images.shape[0] == train_labels.size, \
        'train_images.shape[0] != train_labels.size'
    
    if config['permute_labels']:
        np.random.seed(config['seed'])
        train_labels = np.random.permutation(train_labels)

    # add "channel"-dimension
    train_images = np.expand_dims(
        a=train_images,
        axis=1
    )

    return train_images, train_labels


def train_run(
    config,
    images,
    labels,
    wandb_run=None,
    percentage_validation: float=0.05,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    config["seed"] = config["seed"] + config["run"] + config["fold"]
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    config["input_shape"] = images[0].shape
    config["n_train_images"] = images.shape[0]
    idx = np.arange(config["n_train_images"])
    if config["num_folds"]>1:
        config["n_validation_images"] = int(config["n_train_images"]/config["num_folds"])
        config["validation_idx"] = idx[(config["n_validation_images"]*config["fold"]):(config["n_validation_images"]*(config["fold"]+1))]
    else:
        print(
            f'! Using {percentage_validation*100}% of trainnig data (N: {config["n_train_images"]}) for validation'
        )
        config["validation_idx"] = np.random.choice(
            idx,
            max(1, int(config["n_train_images"]*percentage_validation)),
            replace=False
        )
    train_idx = np.array([i for i in idx if i not in config["validation_idx"]])
    assert np.all([i not in train_idx for i in config["validation_idx"]]), \
        'Validation set contains training images'
    tensor_train_images = torch.Tensor(images[train_idx]).to(torch.float)
    tensor_train_labels = torch.Tensor(labels[train_idx]).to(torch.long)
    tensor_validation_images = torch.Tensor(images[config["validation_idx"]]).to(torch.float)
    tensor_validation_labels = torch.Tensor(labels[config["validation_idx"]]).to(torch.long)

    if 'cuda' in config['device']:    
        tensor_train_images = tensor_train_images.to(config['device'])
        tensor_train_labels = tensor_train_labels.to(config['device'])
        tensor_validation_images = tensor_validation_images.to(config['device'])
        tensor_validation_labels = tensor_validation_labels.to(config['device'])
    
    train_dataset = torch.utils.data.TensorDataset(
        tensor_train_images,
        tensor_train_labels
    )
    validation_dataset = torch.utils.data.TensorDataset(
        tensor_validation_images,
        tensor_validation_labels
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False
    )    
    model = src.model.CNNModel(
        input_shape=config["input_shape"],
        num_classes=np.unique(labels).size,
        num_filters=config["num_filters"],
        filter_size=config["filter_size"],
        num_hidden_layers=config["num_hidden_layers"],
        dropout=config["dropout"]
    )
    
    if 'cuda' in config['device']:
        model.to(config['device'])

    xe_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"]
    )
    train_history = []
    validation_history = []

    for epoch in range(config["num_epochs"]):
        model.train()
        train_losses = []
        train_accuracies = [] 

        for batch_images, batch_labels in iter(train_dataloader):
            optimizer.zero_grad()
            batch_outputs = model(batch_images)
            batch_loss = xe_loss(batch_outputs, batch_labels)
            batch_loss.backward()
            optimizer.step()
            batch_acc = np.mean(
                batch_outputs.detach().cpu().numpy().argmax(axis=1).ravel()==
                batch_labels.cpu().numpy().ravel()
            )
            train_losses.append(batch_loss.detach().cpu().numpy())
            train_accuracies.append(batch_acc)
        
        train_history.append(
            pd.DataFrame(
                {
                    'run': config["run"],
                    'fold': config["fold"],
                    'epoch': epoch,
                    'loss': np.mean(train_losses),
                    'accuracy': np.mean(train_accuracies)
                },
                index=[epoch]
            )
        )
        
        model.eval()
        
        with torch.no_grad():
            eval_losses = []
            eval_accuracies = []

            for image, label in iter(validation_dataloader):
                sample_output = model(image)
                sample_loss = xe_loss(sample_output, label)
                sample_acc = int(
                    sample_output.detach().cpu().numpy().argmax(axis=1)==
                    label.cpu().numpy()
                )
                eval_losses.append(sample_loss.detach().cpu().numpy())
                eval_accuracies.append(sample_acc)
        
        validation_history.append(
            pd.DataFrame(
                {
                    'run': config["run"],
                    'fold': config["fold"],
                    'epoch': epoch,
                    'loss': np.mean(eval_losses),
                    'accuracy': np.mean(eval_accuracies)
                },
                index=[epoch]
            )
        )
        
        if config["report_to"] == "wandb":
             wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_history[-1]['loss'].values[0],
                    "train_acc": train_history[-1]['accuracy'].values[0],
                    "eval_loss": validation_history[-1]['loss'].values[0],
                    "eval_acc": validation_history[-1]['accuracy'].values[0],
                }
            )
        
        if config["verbose"]:
            print(
                f'\t\tEpoch: {epoch}; '
                f'Train (loss, acc): '
                f'{train_history[-1]["loss"].values[0]:.4f}, '
                f'{train_history[-1]["accuracy"].values[0]:.4f}; '
                f'Eval (loss, acc): '
                f'{validation_history[-1]["loss"].values[0]:.4f}, '
                f'{validation_history[-1]["accuracy"].values[0]:.4f}'
            )

    torch.save(
        model.state_dict(),
        os.path.join(
            config["run_log_dir"],
            'final_model.pt'
        )
    )
    run_config_path = os.path.join(
        config["run_log_dir"],
        'config.json'
    )
    config['validation_idx'] = config['validation_idx'].tolist()
    with open(run_config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return (
        pd.concat(train_history, ignore_index=True),
        pd.concat(validation_history, ignore_index=True)
    )


if __name__ == '__main__':
    train()