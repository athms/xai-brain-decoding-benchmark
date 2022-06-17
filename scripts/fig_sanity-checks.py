#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(
    context='paper',
    style='ticks'
)


def fig_sanity_checks(config=None) -> None:
    """Script's main function; creates overview figure
    for results of sanity checks analysis (scripts/sanity-checks_analysis.py)"""
    
    if config is None:
        config = vars(get_argparse().parse_args())

    os.makedirs(
        config["figures_dir"],
        exist_ok=True
    )

    fig, fig_axs = plt.subplot_mosaic(
        """
        AB
        CD
        EF
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
                [fig_axs['A'],fig_axs['B']],
                [fig_axs['C'],fig_axs['D']],
                [fig_axs['E'],fig_axs['F']]
            ]
        )
    ):

        randomized_label_data = pd.read_csv(
            os.path.join(
                config["sanity_checks_base_dir"],
                f"task-{task}",
                "sanity-check_randomized-labels.csv"
            )
        )

        if task_i == 0:
            axs[0].set_title('Data randomization')

        sns.boxenplot(
            data=randomized_label_data,
            x="method",
            y="mi",
            ax=axs[0],
            order=method_ordering,
            palette=sns.color_palette("Paired"),
            scale='linear',
            linewidth=0.5
        )
        sns.despine(ax=axs[0])
        axs[0].set_xlabel('')
        ylabel=r'$MI: Attr_{Data}, Attr_{Rand}$'
        axs[0].set_ylabel(f'{task}\n\n{ylabel}')
        axs[0].set_ylim(0, 0.5)

        if task_i == 2:
            ticklabels = axs[0].get_xticklabels()    
            axs[0].set_xticklabels(
                ticklabels,
                rotation=45,
                fontweight='light',
                horizontalalignment='right'
            )

        else:
            axs[0].set_xticklabels([])

        randomized_model_data = pd.read_csv(
            os.path.join(
                config["sanity_checks_base_dir"],
                f"task-{task}",
                "sanity-check_randomized-model.csv"
            )
        )

        if task_i == 0:
            axs[1].set_title('Model parameter randomization')

        sns.boxenplot(
            data=randomized_model_data,
            x="method",
            y="mi",
            ax=axs[1],
            order=method_ordering,
            palette=sns.color_palette("Paired"),
            scale='linear',
            linewidth=0.5
        )
        sns.despine(ax=axs[1])  
        axs[1].set_xlabel('') 
        axs[1].set_ylabel(r'$MI: Attr_{Data}, Attr_{Rand}$')
        axs[1].set_ylim(0, 0.5)

        if task_i == 2:
            ticklabels = axs[1].get_xticklabels()    
            axs[1].set_xticklabels(
                ticklabels,
                rotation=45,
                fontweight='light',
                horizontalalignment='right'
            )

        else:
            axs[1].set_xticklabels([])

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
    fig.savefig(
        fname=os.path.join(
            config["figures_dir"],
            'Fig-5_sanity-checks.png'
        ),
        dpi=300
    )


def get_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='overview figure for sanity checks analysis'
    )
    parser.add_argument(
        '--figures-dir',
        metavar='DIR',
        default='figures/',
        type=str,
        required=False,
        help='path where figures are stored (default: figures/)'
    )
    parser.add_argument(
        '--sanity-checks-base-dir',
        metavar='DIR',
        default='results/sanity_checks',
        type=str,
        required=False,
        help='path to base directory where results for sanity checks analysis '
             'are stored for all tasks (default: results/sanity_checks)'
    )
    return parser


if __name__ == '__main__':
    
    fig_sanity_checks()