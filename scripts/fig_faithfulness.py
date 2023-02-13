#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import bambi as bmb
import arviz as az
from sqlite3 import DataError
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
sns.set_theme(
    context='paper',
    style='ticks'
)


def test_hdi(hdi: np.ndarray, true_value: float = 0) -> bool:
    """Test whether true value is in HDI"""
    return hdi[0] <= true_value <= hdi[1]


def fig_faithfulness(config=None) -> None:
    """Script's main function; creates overview
    figure for faithfulness analysis (scripts/faithfulness_analysis.py)."""

    if config is None:
        config = vars(get_argparse().parse_args())

    np.random.seed(config['seed'])

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
        'InputXGradient',
        'Gradient',
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
        chance_first_reached['task'] = task
        # mfx_data.append(chance_first_reached)
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
        axs[0].set_xlim(0, 40)
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
            alpha=0.75,
            edgecolor='gray',
            linewidth=0.5,
            ax=axs[1],
            palette=plotting_colors,
            size=4,
            zorder=-1
        )
        axs[1].legend_.remove()

        if task_i == 0:
            axs[1].set_title('Chance-level reached')

        else:
            axs[1].set_title('')

        axs[1].set_ylim(0, None)
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

        # compute mixed effects model
        mfx_results_path = os.path.join(
            config["mfx_dir"],
            f'faithfulness-{task}_mfx-results.csv'
        )
        if not os.path.isfile(mfx_results_path):
            chance_first_reached = chance_first_reached[chance_first_reached['method'].isin(method_ordering)]
            chance_first_reached = pd.get_dummies(chance_first_reached, columns=['method'])
            colname_mapper = {
                c: c.split('method_')[1]
                for c in chance_first_reached.columns
                if 'method' in c
            }
            chance_first_reached.rename(columns=colname_mapper, inplace=True)
            chance_first_reached['DeepLift'] = 0
            fixed_effects = " + ".join([m for m in method_ordering if m!='DeepLift']) # DeepLift is the reference method
            model_string = f"fraction_occluded ~ {fixed_effects}"
            print(
                f'\nComputing regression model:\n\t{model_string}\n'
            )
            mfx_model = bmb.Model(model_string, chance_first_reached)

            n_tune = 5000
            converged = False
            while not converged:
                results = mfx_model.fit(
                    draws=5000,
                    tune=n_tune,
                    chains=4,
                    random_seed=config['seed']
                )
                mfx_result = az.summary(results)

                if all(np.abs(mfx_result['r_hat']-1) < .05):
                    converged = True
                
                n_tune += 5000

            az.plot_trace(results);
            plt.tight_layout()
            os.makedirs(
                config["mfx_dir"],
                exist_ok=True
            )
            plt.savefig(
                fname=os.path.join(
                    config["mfx_dir"],
                    f'faithfulness-{task}_mfx-trace.png'
                ),
                dpi=300
            )
            mfx_result.to_csv(mfx_results_path)

        else:
            mfx_result = pd.read_csv(mfx_results_path, index_col=0)

        # plot indicator for meaningful differences
        for mi, method in enumerate(method_ordering):

            if method != 'DeepLift':
            
                if not test_hdi(mfx_result.loc[method, ['hdi_3%','hdi_97%']]):
                    axs[1].text(
                        s='*',
                        x=mi,
                        y=float(axs[1].get_ylim()[1])*0.9,
                        ha='center',
                        va='bottom',
                        fontsize=20
                    )

        axs[1].set_ylim(0,  float(axs[1].get_ylim()[1])*1.15)

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
            'Fig-4_faithfulness.png'
        ),
        dpi=300
    )

    return fig


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
    parser.add_argument(
        '--mfx-dir',
        metavar='DIR',
        default='results/mfx',
        type=str,
        required=False,
        help='path where results of mixed effects analysis are stored '
             '(default: results/mfx)'
    )
    parser.add_argument(
        "--seed",
        type=str,
        metavar='INT',
        default=12345,
        help='random seed'
    )

    return parser


if __name__ == '__main__':
    
    fig_faithfulness();