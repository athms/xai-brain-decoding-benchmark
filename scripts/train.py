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


def train(config: Dict=None) -> None:
    """Script's main function; trains 3D convolutional neural network."""

    config = make_config(config=config)
    
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


def load_train_data(config: Dict):
    """Loads training data, given config."""

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
        test_subjects = subjects[::5] # use every 5th subject for testing
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


class EarlyStopping:
    """Early stopping criterion that stops training when loss does not 
    improve for given number of parience epochs.
    
    Parameters
    ----
    patience : int
        Number of patience epochs
    min_delta : float
        minimum difference in loss that counts as meaningful change
    grace_period : int
        minimum number of timesteps before a training can be early stopped
    """

    def __init__(self,
        patience: int=3,
        min_delta: float=0.0,
        grace_period: int=10
        ) -> None:

        self.patience = patience
        self.min_delta = min_delta
        self.grace_period = grace_period
        self.min_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss: float, epoch: int):
        
        if epoch >= self.grace_period:

            if (loss - self.min_loss) >= self.min_delta:
                self.counter +=1
                if self.counter >= self.patience:  
                    self.early_stop = True

        if loss < self.min_loss:
            self.min_loss = loss



def train_run(
    config,
    images,
    labels,
    wandb_run = None,
    percentage_validation: float=0.05,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trains 3D convolutional neural network, given config, images, and labels.

    Parameters
    ----------
    config : Dict
        Dictionary of configuration parameters.
    images : np.ndarray
        Array of images.
    labels : np.ndarray
        Array of labels.
    wandb_run : wandb run object
        Wandb run object used for logging.
    percentage_validation : float
        Percentage of data to use for validation.

    Returns
    -------
    train_history : pd.DataFrame
        Training history.
    validation_history : pd.DataFrame
        Validation history.
    """
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
            f'\t! Using {percentage_validation*100}% of trainnig data (Total N: {config["n_train_images"]}) for validation'
        )
        config["validation_idx"] = np.random.choice(
            idx,
            max(1, int(config["n_train_images"]*percentage_validation)),
            replace=False
        )
    
    train_idx = np.array([i for i in idx if i not in config["validation_idx"]])
    assert np.all([i not in train_idx for i in config["validation_idx"]]), \
        'Validation set contains training images'
    train_images = torch.Tensor(images[train_idx]).to(torch.float)
    train_labels = torch.Tensor(labels[train_idx]).to(torch.long)
    validation_images = torch.Tensor(images[config["validation_idx"]]).to(torch.float)
    validation_labels = torch.Tensor(labels[config["validation_idx"]]).to(torch.long)

    if 'cuda' in config['device']:    
        train_images = train_images.to(config['device'])
        train_labels = train_labels.to(config['device'])
        validation_images = validation_images.to(config['device'])
        validation_labels = validation_labels.to(config['device'])
    
    train_dataset = torch.utils.data.TensorDataset(
        train_images,
        train_labels
    )
    validation_dataset = torch.utils.data.TensorDataset(
        validation_images,
        validation_labels
    )

    # adapted from: https://pytorch.org/docs/stable/notes/randomness.html
    def seed_worker(worker_id):
        """helper function to correctly seed dataloader workers"""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)

    train_gen = torch.Generator()
    train_gen.manual_seed(config["seed"])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=train_gen,
        worker_init_fn=seed_worker
    )
    val_gen = torch.Generator()
    val_gen.manual_seed(config["seed"])
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        generator=val_gen,
        worker_init_fn=seed_worker
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
    earl_stopping = EarlyStopping()

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

        earl_stopping(loss=np.mean(eval_losses), epoch=epoch)
        if earl_stopping.early_stop:
            print(
                'Stopping training as early-stopping criterion reached.'
            )
            break

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


def get_train_argparse(parser: argparse.ArgumentParser=None) -> argparse.ArgumentParser:
    
    if parser is None:
        parser = argparse.ArgumentParser(
            description='train 3D-CNN for given task'
        )

    parser.add_argument(
        '--task',
        metavar='TASK',
        default='WM',
        type=str,
        help='task for which 3D-CNN is trained '
             '(default: WM)'
    )
    parser.add_argument(
        '--data-dir',
        metavar='DIR',
        default='data/task-WM',
        type=str,
        required=False,
        help='path where trial-level BOLD GLM maps are stored '
             '(default: data/)'
    )


    parser.add_argument(
        '--num-hidden-layers',
        metavar='N',
        default=4,
        type=int,
        required=False,
        help='number of hidden 3D-CNN layers '
             '(default: 4)'
    )
    parser.add_argument(
        '--num-filters',
        metavar='N',
        default=16,
        type=int,
        required=False,
        help='number of 3D-CNN kernels per layer '
             '(default: 8)'
    )
    parser.add_argument(
        '--filter-size',
        metavar='N',
        default=4,
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
             '(default: 40)'
    )
    parser.add_argument(
      '--learning-rate',
      metavar='FLOAT',
      default=3e-4,
      type=float,
      required=False,
      help='learning rate for AdamW '
           '(default: 3e-4)'
    )
    parser.add_argument(
      '--dropout',
      metavar='FLOAT',
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
      help='Number of CV-folds per training run'
           '(default: 3)'
    )
    parser.add_argument(
      '--run',
      metavar='INT',
      default=-1,
      type=int,
      required=False,
      help='run to compute (overwrites num-runs) '
           '(default: -1)'
    )
    parser.add_argument(
      '--fold',
      metavar='INT',
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
        help='path where models and logs are stored '
             '(default: results/models/)'
    )
    parser.add_argument(
        '--run-group-name',
        metavar='STR',
        default='none',
        type=str,
        required=False,
        help='group name for runs used during logging '
             '(default: none)'
    )
    parser.add_argument(
        '--report-to',
        metavar='STR',
        default='none',
        type=str,
        choices=('none', 'wandb', 'tune'),
        required=False,
        help='whether to report training results to wandb, tune or none '
             '(default: none)'
    )
    parser.add_argument(
        '--wandb-entity',
        metavar='STR',
        type=str,
        default='anonymous',
        required=False,
        help='entity used for wandb logging'
             'only needed if report-to is wandb'
             '(default: anonymous)'
    )
    parser.add_argument(
        '--wandb-project',
        metavar='STR',
        default='interpreting-brain-decoding-models',
        type=str,
        required=False,
        help='project used for wandb logging '
             'only needed if report-to is wandb'
             '(default: interpreting-brain-decoding-models)'
    )
    parser.add_argument(
        '--wandb-mode',
        metavar='STR',
        default='online',
        choices=('online', 'offline', 'disabled'),
        type=str,
        required=False,
        help='mode used for wandb logging '
             'only needed if report-to is wandb'
             'one of [online, offline, disabled] (default: online)'
    )
    parser.add_argument(
        '--smoke-test',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        required=False,
        help='whether to run training in smoke-test mode '
             'used for testing purposes only (default: False)'
    )
    parser.add_argument(
        '--verbose',
        metavar='BOOL',
        default='True',
        choices=('True', 'False'),
        type=str,
        required=False,
        help='actively comment training? '
             '(default: True)'
    )
    parser.add_argument(
        '--seed',
        metavar='INT',
        default=12345,
        type=int,
        required=False,
        help='initial random seed '
             '(default: 1234)'
    )
    parser.add_argument(
        '--permute-labels',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        required=False,
        help='whether or not to permute training labels '
             'if permuted, attributions only computed for one fitting run '
             '(default: False)'
    )
    parser.add_argument(
        '--model-config',
        metavar='PATH',
        default='none',
        type=str,
        required=False,
        help='path to model configuration file used to specify model '
             'if specified, overwrites all other script settings for the model!'
             '(default: none)'
    )

    return parser


def make_config(config: Dict=None):
    """Generates config dictionary"""
    
    if config is None:
        config = vars(get_train_argparse().parse_args())
    
    config["verbose"] = config["verbose"] == 'True'
    config["permute_labels"] = config["permute_labels"] == 'True'
    config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    config["smoke_test"] = config["smoke_test"] == 'True'
    
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
            'dropout',
            'num_epochs'
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


if __name__ == '__main__':

    train()