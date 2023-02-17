#!/usr/bin/env python3

import os
import argparse
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(
    context='paper',
    style='ticks'
)


def randomized_labels_performance(config: Dict=None) -> None:
    """Script's main function; plots performance of models
    trained on variant of the data with randomized labels."""

    if config is None:
        config =  vars(get_argsparse().parse_args())

    fig, fig_axs = plt.subplot_mosaic(
        """
        AB
        CD
        EF
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
                [fig_axs['A'], fig_axs['B']],
                [fig_axs['C'], fig_axs['D']],
                [fig_axs['E'], fig_axs['F']]
            ]
        )
    ):
        print(f"Processing: {task}")
        model_dir = [
            os.path.join(config['model_dir'], p)
            for p in os.listdir(config['model_dir'])
            if p.startswith(f"task-{task}")
        ]
        assert len(model_dir) == 1, f"too many models ({len(model_dir)}) found for task {task}"
        model_dir = model_dir[0]

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
        for history_i, (label, history) in enumerate(
            zip(
                ['training', 'validation'],
                [train_history, val_history]
            )
        ):
            axs[history_i].plot(
                history['epoch'].values,
                history['accuracy'].values*100,
                color=sns.color_palette('Set3')[-(history_i+1)],
                zorder=history_i,
                lw=2,
            )
            epochs = np.sort(history['epoch'].unique())
            axs[history_i].set_ylim(0, 100)
            axs[history_i].set_xlim(0, np.max(epochs))
            xticks = np.insert(epochs[::250], -1, epochs[-1]+1)
            axs[history_i].set_xticks(xticks)
            axs[history_i].set_xticklabels([e if e%250==0 else '' for e in xticks])
            axs[history_i].set_xlabel('Epoch')
            print(
                f"\tFinal {label} decoding accuracy:{history['accuracy'].values[-1]*100}"
            )

        _ = [sns.despine(ax=ax) for ax in axs]

        if task_i == 0:
            axs[0].set_title('Training')
            axs[1].set_title('Validation')

        axs[0].set_ylabel(f'{task}\n\nAccuracy (%)')
        axs[1].set_ylabel('')

    for label in list('ABCDEF'):
        fig_axs[label].text(
            -0.1,
            1.25,
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
            f'Sfig_randomized-labels-performance.png'
        ),
        dpi=300
    )


def get_argsparse(parser: argparse.ArgumentParser=None) -> argparse.ArgumentParser:
    
    if parser is None:
        parser = argparse.ArgumentParser(
            description='get best configs for given task',
        )

    parser.add_argument(
        '--model-dir',
        metavar='DIR',
        type=str,
        help='',
        default='results/models/randomized_labels'
    )
    parser.add_argument(
        '--figures-dir',
        metavar='DIR',
        default='figures',
        type=str,
        help=''
    )
    
    return parser


if __name__ == '__main__':
    randomized_labels_performance()