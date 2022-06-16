#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
from sqlite3 import DataError
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
sns.set_theme(
    context='paper',
    style='ticks'
)


def fig_faithfulness(config=None) -> None:
    """Script's main function; creates overview
    figure for faithfulness analysis (scripts/faithfulness_analysis.py)."""

    if config is None:
        config = vars(get_argparse().parse_args())

    fig, fig_axs = plt.subplot_mosaic(
        """
        AAABB
        CCCDD
        EEEFF
        """,
        figsize=(8, 6),
        dpi=300
    )
    method_ordering = [
        'DeepLift',
        'DeepLiftShap',
        'IntegratedGradients',
        'LRP',
        'Gradient',
        'InputXGradient',
        'SmoothGrad',
        'GuidedBackprop',
        'GuidedGradCam',
    ]

    for task_i, (task, axs) in enumerate(
        zip(
            ['heat-rejection', 'MOTOR', 'WM'],
            [
                [fig_axs['A'], fig_axs['B']],
                [fig_axs['C'], fig_axs['D']],
                [fig_axs['E'], fig_axs['F']]
            ]
        )
    ):

        faithfulness_path = os.path.join(
            config["faithfulness_base_dir"],
            f'task-{task}',
            'faithfulness_analysis.csv'
        )
        chance_first_reached_path = os.path.join(
            config["faithfulness_base_dir"],
            f'task-{task}',
            'chance_first_reached_analysis.csv'
        )

        if not os.path.isfile(faithfulness_path) or not os.path.isfile(chance_first_reached_path):

            raise DataError(
                f'{faithfulness_path} and {chance_first_reached_path} not found '
                f'Please run scripts/faithfulness_analysis.py for {task} task first'
            )

        faithfulness = pd.read_csv(faithfulness_path)
        chance_first_reached = pd.read_csv(chance_first_reached_path)
        fractions = np.sort(np.unique(faithfulness['fraction_occluded']))
        axs[0].axhline(
            y=chance_first_reached['chance'].mean(),
            color='gray',
            linewidth=1,
            ls='--'
        )
        plotting_colors = []

        for i, attribution_method in enumerate(method_ordering):
            method_faithfulness = faithfulness.loc[faithfulness['method']==attribution_method]
            grouped_faithfulness = method_faithfulness.groupby(['fraction_occluded'])
            mean_accuracies = grouped_faithfulness.accuracy.mean()
            plotting_colors.append(sns.color_palette("Paired")[i])
            axs[0].plot(
                mean_accuracies.index,
                mean_accuracies,
                color=plotting_colors[-1],
                linewidth=0.5
            )
            axs[0].scatter(
                mean_accuracies.index,
                mean_accuracies,
                color=plotting_colors[-1],
                edgecolors='none',
                marker=Line2D.filled_markers[i][0],
                s=12,
                label=attribution_method
            )

        if task_i == 2:
            axs[0].set_xlabel('Occlusion (%)')

        else:
            axs[0].set_xlabel('')

        axs[0].set_xticks(fractions)
        axs[0].set_xticklabels([int(i) if i%1==0 else '' for i in fractions])
        axs[0].set_xlim(0, np.round(fractions.max()))
        axs[0].set_ylabel(f'{task}\n\nTest accuracy (%)')

        if task_i == 1:
            axs[0].legend(
                loc='upper right',
                frameon=True,
                markerscale=2,
                ncol=3,
                fontsize=7
            )

        axs[0].set_yticks(np.arange(0,120,20).astype(np.int))
        axs[0].set_yticklabels([0, 20, 40, 60, 80, 100])
        axs[0].set_ylim(0, 100)
        sns.swarmplot(
            data=chance_first_reached,
            x="method",
            y="fraction_occluded",
            hue='method',
            order=method_ordering,
            hue_order=method_ordering,
            ax=axs[1],
            palette=plotting_colors,
            size=4,
            alpha=0.5,
            zorder=-1
        )
        sns.pointplot(
            data=chance_first_reached,
            x="method",
            y="fraction_occluded",
            hue='method',
            order=method_ordering,
            ax=axs[1],
            color='black',
            size=3,
            markers='x',
            ci=None,
            zorder=99
        )
        axs[1].legend_.remove()

        if task_i == 0:
            axs[1].set_title('Chance-level reached')

        else:
            axs[1].set_title('')

        axs[1].set_ylim(0, max(fractions))
        axs[1].set_ylabel('Occlusion (%)')
        axs[1].set_xlabel('')

        if task_i == 2:
            axs[1].set_xticklabels(
                axs[1].get_xticklabels(),
                rotation=45,
                fontweight='light',
                horizontalalignment='right'
            )

        else:
            axs[1].set_xticklabels([])

        _ = [sns.despine(ax=ax) for ax in axs]

    for label in list('ABCDEF'):
        fig_axs[label].text(
            -0.05,
            1.25,
            label,
            transform=fig_axs[label].transAxes,
            fontsize=12,
            fontweight='bold',
            va='top'
        )

    fig.tight_layout()
    os.makedirs(
        config['figures_dir'],
        exist_ok=True
    )
    fig.savefig(
        fname=os.path.join(
            config['figures_dir'],
            'faithfulness.png'
        ),
        dpi=300
    )


def get_argparse() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description='create overview figure for faithfulness analysis'
    )
    parser.add_argument(
        '--faithfulness-base-dir',
        metavar='DIR',
        default='results/faithfulness',
        type=str,
        required=False,
        help='path to base directory where results of faithfulness analysis '
             'are stored for all tasks '
             '(default: results/faithfulness)'
    )
    parser.add_argument(
        '--figures-dir',
        metavar='DIR',
        default='figures',
        type=str,
        required=False,
        help='path where figures are stored (default: figures)'
    )

    return parser


if __name__ == '__main__':
    
    fig_faithfulness()