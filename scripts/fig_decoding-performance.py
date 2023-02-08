#!/usr/bin/env python3 

import os, sys
import argparse
import json
from typing import Dict
import numpy as np
import pandas as pd
import collections
from nilearn.image import load_img
from sklearn.metrics import confusion_matrix
import torch
sys.path.append('./')
from src import target_labeling
from src.data import get_subject_trial_image_paths
from src.model import CNNModel
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(
    context='paper',
    style='ticks'
)


def fig_decoding_performance(config: Dict=None) -> None:
    """Script's main function; creates overview figure 
    for model decoding performances in each of the three
    datasets (ie., tasks).
    """

    if config is None:
        config =  vars(get_argparse().parse_args())

    os.makedirs(
        config["figures_dir"],
        exist_ok=True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fig, fig_axs = plt.subplot_mosaic(
        """
        AAABBBCCDDD
        EEEFFFGGHHH
        IIIJJJKKLLL
        """,
        figsize=(9, 6),
        dpi=300
    )

    for task_i, (task, axs) in enumerate(
        zip(
            [
                'heat-rejection',
                'MOTOR',
                'WM'
            ],
            [
                [fig_axs['A'], fig_axs['B'], fig_axs['C'], fig_axs['D']],
                [fig_axs['E'], fig_axs['F'], fig_axs['G'], fig_axs['H']],
                [fig_axs['I'], fig_axs['J'], fig_axs['K'], fig_axs['L']]
            ]
        )
    ):

        _ = [sns.despine(ax=ax) for ax in axs]

        model_dir = [
            os.path.join(config['fitted_models_base_dir'], p)
            for p in os.listdir(config['fitted_models_base_dir'])
            if p.startswith(f"task-{task}")
        ]
        assert len(model_dir) == 1, f"too many models ({len(model_dir)}) found for task {task}"
        model_dir = model_dir[0]
        model_config = json.load(open(os.path.join(model_dir, 'config.json')))

        with open(
            os.path.join(
                config['data_base_dir'],
                f'task-{task}',
                'train_test_split.json'
                ),
            'r'
        ) as f:
            trial_image_paths = json.load(f)

        fitting_runs = np.sort(
            a=[
                int(d.split('-')[1])
                for d in os.listdir(model_dir)
                if d.startswith('run-')
            ]
        )
        test_subjects = list(trial_image_paths['test'].keys())
        test_image_paths = get_subject_trial_image_paths(
            path=os.path.join(config["data_base_dir"], f'task-{task}'),
            subjects=test_subjects,
            decoding_targets=target_labeling[task].keys()
        )
        test_data = {
            'image_path': [],
            'label': [],
            'numeric_label': [],
            'image': [],
            'nii_image': []
        }

        for subject in test_image_paths:

            for image_path in test_image_paths[subject]:
                label = image_path.split('/')[-1].split('_')[0]
                nii_image = load_img(image_path)
                image = np.expand_dims(nii_image.get_fdata(), axis=[0, 1])
                test_data['label'].append(label)
                test_data['numeric_label'].append(target_labeling[task][label])
                test_data['image'].append(image)
                test_data['image_path'].append(image_path)
                test_data['nii_image'].append(nii_image)

        chance_acc = max(collections.Counter(test_data['numeric_label']).values())
        chance_acc /= len(test_data['numeric_label'])
        num_labels = len(np.unique(test_data['numeric_label']))
        input_shape = image.shape[1:]

        model = CNNModel(
            input_shape=input_shape,
            num_classes=num_labels,
            num_filters=model_config["num_filters"],
            filter_size=model_config["filter_size"],
            num_hidden_layers=model_config["num_hidden_layers"],
            dropout=model_config["dropout"]
        )

        if task_i == 0:
            axs[0].set_title('Training')
            axs[1].set_title('Validation')

        axs[0].set_ylabel(f'{task}\n\nAccuracy (%)')
        axs[0].axhline(
            y=chance_acc * 100,
            color='gray',
            linewidth=1,
            ls='--'
        )
        axs[1].axhline(
            y=chance_acc * 100,
            color='gray',
            linewidth=1,
            ls='--'
        )
        axs[0].set_xlabel('Epoch')
        axs[1].set_xlabel('Epoch')
        train_history = pd.read_csv(
            os.path.join(
                model_dir,
                'train_history.csv'
            ),
            index_col=False
        )
        val_history = pd.read_csv(
            os.path.join(
                model_dir,
                'validation_history.csv'
            ),
            index_col=False
        )

        # plot training and validation accuracies during training:
        for history_i, (label, history) in enumerate(
            zip(
                ['training', 'validation'],
                [train_history, val_history]
            )
        ):  
            max_epoch = 0
            for run in history['run'].unique():
                history_run = history[history['run']==run].copy()
                axs[history_i].plot(
                    history_run['epoch'],
                    history_run['accuracy']*100,
                    color='gray',
                    alpha=0.5,
                    lw=1,
                )
                if history_run['epoch'].max() > max_epoch:
                    max_epoch = history_run['epoch'].max()
            # history_grouped = history.groupby(['run', 'epoch']).accuracy.mean().groupby('epoch')
            # history_min = history_grouped.min() * 100
            # history_max = history_grouped.max() * 100
            # history_mean = history_grouped.mean() * 100
            # epochs = np.sort(history['epoch'].unique())
            # axs[history_i].plot(
            #     epochs,
            #     history_mean,
            #     color='gray',
            #     zorder=history_i,
            #     lw=1,
            # )
            # axs[history_i].fill_between(
            #     epochs,
            #     history_min,
            #     history_max,
            #     alpha=0.3,
            #     color='gray',
            #     zorder=history_i,
            #     linewidth=0.0
            # )
            axs[history_i].set_ylim(0, 100)
            axs[history_i].set_xlim(0, max_epoch)
            epochs = np.arange(0, max_epoch+5, 5) if max_epoch<31 else np.arange(0, max_epoch+10, 10)
            axs[history_i].set_xticks(epochs)
            axs[history_i].set_xticklabels(epochs)

        # compute average confusion matrix and final decoding accuracies:
        conf_mat = np.zeros((num_labels, num_labels))
        acc = []

        for fitting_run in fitting_runs:
            model_path = os.path.join(
                model_dir,
                f'run-{fitting_run}',
                'best_model.pt'
            )

            if not torch.cuda.is_available():
                model.load_state_dict(
                    torch.load(
                        model_path,
                        map_location=torch.device('cpu')
                    )
                )

            else:
                model.load_state_dict(torch.load(model_path))

            model.eval()

            if torch.cuda.is_available():
                model.to(device)

            test_data['prediction'] = []
            for image in test_data['image']:
                image = torch.from_numpy(image).float()

                if torch.cuda.is_available():
                    image = image.to(device)

                test_data['prediction'].append(
                    model(image).detach().cpu().numpy().argmax(axis=1)[0]
                )

            acc.append(
                np.mean(
                    np.array(test_data['prediction'])==test_data['numeric_label']
                ) * 100
            )
            conf_mat += confusion_matrix(
                y_true=test_data['numeric_label'],
                y_pred=np.array(test_data['prediction']),
                normalize='true'
            ) * 100

        conf_mat /= len(fitting_runs)
        conf_mat = pd.DataFrame(
            conf_mat,
            index=target_labeling[task].keys(),
            columns=target_labeling[task].keys()
        )

        if task_i == 0:
            axs[2].set_title('Test')

        # plot final decoding accuracies:
        axs[2] = sns.swarmplot(
            y=np.array(acc),
            ax=axs[2],
            alpha=0.7,
            color='gray',
            edgecolor='gray',
            linewidth=0.5
        )
        axs[2].set_ylim(0, 100)
        axs[2].set_xticks([0])
        axs[2].set_xticklabels(['Final epoch'])
        axs[2].text(
            0.5,
            0.25,
            f'Mean: {np.mean(acc):.1f}%\n'+\
            f'Min: {np.min(acc):.1f}%\n'+\
            f'Max: {np.max(acc):.1f}%',
            horizontalalignment='center',
            verticalalignment='center',
            transform=axs[2].transAxes,
            size=7,
            color='k'
        )

        if task_i == 0:
            axs[3].set_title('Test\nconfusion (%)')

        # plot average confusion matrix:
        sns.heatmap(
            conf_mat.astype(int),
            vmax=100,
            vmin=0,
            center=50,
            square=True,
            annot=True,
            annot_kws={'size': 8},
            linewidths=.5,
            cbar_kws={
                "shrink": .5,
            },
            ax=axs[3],
        )
        axs[3].set_ylabel('True state')
        yticklabels = [
            k if k!='rejection' else 'reject.'
            for k in target_labeling[task].keys()
        ]
        axs[3].set_yticklabels(
            yticklabels,
            rotation=0,
            fontweight='light'
        )
        axs[3].set_xticklabels(
            yticklabels,
            rotation=45,
            fontweight='light',
            horizontalalignment='right'            
        )

        if task_i == 2:
            axs[3].set_xlabel('Predicted state')

    for label in list('ABCDEFGHIJKL'):
        fig_axs[label].text(
            -0.2,
            1.2,
            label,
            transform=fig_axs[label].transAxes,
            fontsize=12,
            fontweight='bold',
            va='top'
        )

    fig.tight_layout()
    fig.savefig(
        os.path.join(
            config['figures_dir'],
            f'Fig-1_decoding-performance.png'
        ),
        dpi=300
    )


def get_argparse(parser: argparse.ArgumentParser=None) -> argparse.ArgumentParser:
    
    if parser is None:
        parser = argparse.ArgumentParser(
            description='create decoding performance figure',
        )

    parser.add_argument(
        '--fitted-models-base-dir',
        metavar='DIR',
        type=str,
        default='results/models',
        help='directory where final model fitting runs are stored '
             'for each task (default: results/models)'
    )
    parser.add_argument(
        '--data-base-dir',
        metavar='DIR',
        type=str,
        default='data',
        help='path to base directory where trial-level GLM maps are stored '
             'for all tasks (default: data/)'
    )
    parser.add_argument(
        '--figures-dir',
        metavar='DIR',
        default='figures',
        type=str,
        help='path where figure is saved (default: figures)'
    )
    
    return parser


if __name__ == '__main__':
    
    fig_decoding_performance()