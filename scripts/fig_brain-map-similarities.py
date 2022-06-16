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


def fig_brain_maps_similarity(config=None) -> None:
    """Script's main function; creates overview figure
    for results of running scripts/brain-map-similaritues_analysis.py"""
    
    if config is None:
        config = vars(get_argparse().parse_args())

    os.makedirs(
        config["figures_dir"],
        exist_ok=True
    )

    fig, fig_axs = plt.subplot_mosaic(
        """
        ABB
        CDD
        EFF
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

        for ai, analysis_level in enumerate(['group', 'subject']):
            ax = axs[ai]
            brain_map_similarities = pd.read_csv(
                os.path.join(
                    config["brain_maps_similarity_base_dir"],
                    f'task-{task}',
                    f'brain_map_similarities_{analysis_level}.csv'
                )
            )
            
            if len(brain_map_similarities['contrast'].unique()) == 2:
                brain_map_similarities = brain_map_similarities[brain_map_similarities['contrast']==brain_map_similarities['contrast'].unique()[0]]

            if analysis_level in {'group'}:
                sns.barplot(
                    data=brain_map_similarities,
                    x="method",
                    y="mi",
                    ax=ax,
                    order=method_ordering,
                    palette=sns.color_palette("Paired"),
                    ci=None,
                )
            
            elif analysis_level == 'subject':
                sns.boxenplot(
                    data=brain_map_similarities,
                    x="method",
                    y="mi",
                    ax=ax,
                    order=method_ordering,
                    palette=sns.color_palette("Paired"),
                    scale='linear',
                    linewidth=0.5
                )
            
            else:
                raise ValueError(
                    f'Unknown analysis level: {analysis_level}'
                )
            
            if task_i == 0:
                ax.set_title(f"{analysis_level.capitalize()} maps")
            
            ax.set_xlabel('')
            
            if analysis_level == 'group':
                ylabel = r'$MI: Attr_{Group}, BOLD_{Group}$'
            
            elif analysis_level == 'subject':
                ylabel = r'$MI: Attr_{Sub}, BOLD_{Group}$'
            
            else:
                raise ValueError(
                    f'Unknown analysis level: {analysis_level}'
                )
            
            if ai == 0:
                ylabel = f'{task}\n\n{ylabel}'

            ax.set_ylabel(ylabel)
            
            if analysis_level == 'group':
                ax.set_ylim(0, None)
            
            elif analysis_level == 'subject':
                ax.set_ylim(0, None)
            
            else:
                raise ValueError(
                    f'Unknown analysis level: {analysis_level}'
                )
            
            ticklabels = ax.get_xticklabels()
            sns.despine(ax=ax)
            
            if task_i == 2:
                ax.set_xticklabels(
                    ticklabels,
                    rotation=45,
                    fontweight='light',
                    horizontalalignment='right'
                )
            
            else:
                ax.set_xticklabels([])
        
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
            f'brain-maps-similarity.png'
        ),
        dpi=300
    )


def get_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='create overview figure for brain maps similarity analysis'
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
        '--brain-maps-similarity-base-dir',
        metavar='DIR',
        default='results/brain_map_similarity',
        type=str,
        required=False,
        help='path to directory where data of brain maps similarity '
             'analysis are stored for all tasks are stored '
             '(default: results/brain_map_similarity)'
    )
    return parser


if __name__ == '__main__':
    
    fig_brain_maps_similarity()