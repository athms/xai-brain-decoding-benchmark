#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bambi as bmb
import arviz as az
import seaborn as sns
sns.set_theme(
    context='paper',
    style='ticks'
)


def test_hdi(hdi: np.ndarray, true_value: float = 0) -> bool:
    """Test whether true value is in HDI"""
    return hdi[0] <= true_value <= hdi[1]


def fig_sanity_checks(config=None) -> None:
    """Script's main function; creates overview figure
    for results of sanity checks analysis (scripts/sanity-checks_analysis.py)"""
    
    if config is None:
        config = vars(get_argparse().parse_args())

    np.random.seed(config['seed'])

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
        'InputXGradient',
        'Gradient',
        'SmoothGrad',
        'GuidedBackprop',
        'GuidedGradCam',
    ]

    # collect data for mixed effects model
    mfx_data = {
        'randomized_data': [],
        'randomized_model': []
    }

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
            axs[0].set_title('Randomized data')

        randomized_label_means = randomized_label_data.groupby(
            ["method", "fitting_run", "subject"]
        ).mi.mean().groupby(
            ["method", "fitting_run"]
        ).mean().reset_index()
        mfx_data['randomized_data'].append(randomized_label_means)
        sns.swarmplot(
            data=randomized_label_means,
            x="method",
            y="mi",
            ax=axs[0],
            order=method_ordering,
            alpha=0.75,
            size=4,
            edgecolor='gray',
            linewidth=0.5,
            palette=sns.color_palette("Paired"),
        )
        sns.despine(ax=axs[0])
        axs[0].set_xlabel('')
        ylabel=r'$I(Attr_{Orig}; Attr_{Rand})$'
        axs[0].set_ylabel(f'{task}\n\n{ylabel}')
        axs[0].set_ylim(0.45, -0.1)
        axs[0].axhline(0, lw=1, ls='--', color='gray')

        # compute mixed effects model
        mfx_results_path = os.path.join(
            config["mfx_dir"],
            f'sanity-checks-random-data-{task}_mfx-results.csv'
        )
        if not os.path.isfile(mfx_results_path):
            randomized_label_means = randomized_label_means[randomized_label_means['method'].isin(method_ordering)]
            randomized_label_means = pd.get_dummies(randomized_label_means, columns=['method'])
            colname_mapper = {
                c: c.split('method_')[1]
                for c in randomized_label_means.columns
                if 'method' in c
            }
            randomized_label_means.rename(columns=colname_mapper, inplace=True)
            randomized_label_means['DeepLift'] = 0
            fixed_effects = " + ".join([m for m in method_ordering if m!='DeepLift']) # DeepLift is the reference method
            model_string = f"mi ~ {fixed_effects}"
            print(
                f'\nComputing regression model:\n\t{model_string}\n'
            )
            mfx_model = bmb.Model(model_string, randomized_label_means)

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
            plt.savefig(
                fname=os.path.join(
                    config["mfx_dir"],
                    f'sanity-checks-random-data-{task}_mfx-trace.png'
                ),
                dpi=300
            )
            mfx_result.to_csv(mfx_results_path)

        else:
            mfx_result = pd.read_csv(mfx_results_path, index_col=0)

        # plot indicator for meaningful differences
        for mi, method in enumerate(method_ordering):

            if method != 'DeepLift':
            
                if 0 > mfx_result.loc[method, 'hdi_97%']:
                    axs[0].text(
                        s='*',
                        x=mi,
                        y=0.05, #float(axs[0].get_ylim()[1])*0.9,
                        ha='center',
                        va='bottom',
                        fontsize=20
                    )

        #axs[0].set_ylim(float(axs[0].get_ylim()[1])*1.15, 0)

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
            axs[1].set_title('Randomized model')

        randomized_model_means = randomized_model_data.groupby(
            ["method", "fitting_run", "subject"]
        ).mi.mean().groupby(
            ["method", "fitting_run"]
        ).mean().reset_index()
        mfx_data['randomized_model'].append(randomized_model_means)
        sns.swarmplot(
            data=randomized_model_means,
            x="method",
            y="mi",
            ax=axs[1],
            order=method_ordering,
            alpha=0.75,
            size=4,
            edgecolor='gray',
            linewidth=0.5,
            palette=sns.color_palette("Paired"),
        )
        sns.despine(ax=axs[1])  
        axs[1].set_xlabel('') 
        axs[1].set_ylabel(r'$I(Attr_{Orig}; Attr_{Rand})$')
        axs[1].set_ylim(0.45, -0.1)
        axs[1].axhline(0, lw=1, ls='--', color='gray')

        # compute mixed effects model
        mfx_results_path = os.path.join(
            config["mfx_dir"],
            f'sanity-checks-random-model-{task}_mfx-results.csv'
        )
        if not os.path.isfile(mfx_results_path):
            randomized_model_means = randomized_model_means[randomized_model_means['method'].isin(method_ordering)]
            randomized_model_means = pd.get_dummies(randomized_model_means, columns=['method'])
            colname_mapper = {
                c: c.split('method_')[1]
                for c in randomized_model_means.columns
                if 'method' in c
            }
            randomized_model_means.rename(columns=colname_mapper, inplace=True)
            randomized_model_means['DeepLift'] = 0
            fixed_effects = " + ".join([m for m in method_ordering if m!='DeepLift']) # DeepLift is the reference method
            model_string = f"mi ~ {fixed_effects}"
            print(
                f'\nComputing regression model:\n\t{model_string}\n'
            )
            mfx_model = bmb.Model(model_string, randomized_model_means)

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
            plt.savefig(
                fname=os.path.join(
                    config["mfx_dir"],
                    f'sanity-checks-random-model-{task}_mfx-trace.png'
                ),
                dpi=300
            )
            mfx_result.to_csv(mfx_results_path)

        else:
            mfx_result = pd.read_csv(mfx_results_path, index_col=0)

        # plot indicator for meaningful differences
        for mi, method in enumerate(method_ordering):

            if method != 'DeepLift':
            
                if 0 > mfx_result.loc[method, 'hdi_97%']:
                    axs[1].text(
                        s='*',
                        x=mi,
                        y=0.05, #float(axs[1].get_ylim()[1])*0.9,
                        ha='center',
                        va='bottom',
                        fontsize=20
                    )

        #axs[1].set_ylim(0,  float(axs[1].get_ylim()[1])*1.15)

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
            'Fig-5_sanity-checks.jpg'
        ),
        dpi=300
    )

    return fig


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
        '--mfx-dir',
        metavar='DIR',
        default='results/mfx',
        type=str,
        required=False,
        help='path where results of mixed effects analysis are stored '
             '(default: results/mfx)'
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
    parser.add_argument(
        "--seed",
        type=str,
        metavar='INT',
        default=12345,
        help='random seed'
    )
    return parser


if __name__ == '__main__':
    fig_sanity_checks()