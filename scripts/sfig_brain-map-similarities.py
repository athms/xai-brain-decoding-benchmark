#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import bambi as bmb
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(
    context='paper',
    style='ticks'
)


def sfig_brain_maps_similarity(config=None) -> None:
    """Script's main function; creates overview figure
    for results of running vscripts/brain-map-similaritues_analysis.py"""
    
    if config is None:
        config = vars(get_argparse().parse_args())

    np.random.seed(config['seed'])

    os.makedirs(
        config["figures_dir"],
        exist_ok=True
    )
    os.makedirs(
        config["mfx_dir"],
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
        'InputXGradient',
        'Gradient',
        'SmoothGrad',
        'GuidedBackprop',
        'GuidedGradCam',
    ]

    # collect data for mixed effects model
    mfx_data = []

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
                    f'brain-map-similarities_{analysis_level}.csv'
                )
            )
            
            # if only two contrasts present in the data, we can reduce the data to onle one contrast,
            # as the two contrasts are fully symmetric
            if len(brain_map_similarities['contrast'].unique()) == 2:
                brain_map_similarities = brain_map_similarities[
                    brain_map_similarities['contrast']==brain_map_similarities['contrast'].unique()[0]]

            # create df for seaborn plotting
            if analysis_level=='subject':
                subjects = [im.split('sub_')[1].split('/')[0] for im in 
                    brain_map_similarities['attribution_image']] 
            else:
                subjects = None
            plotting_df = pd.DataFrame({
                'method': brain_map_similarities['method'],
                'r': brain_map_similarities['r_bold'],
                'subject': subjects
            })

            if analysis_level in {'group'}:
                ax = sns.barplot(
                    data=plotting_df,
                    x="method",
                    y='r',
                    ax=ax,
                    order=method_ordering,
                    palette=sns.color_palette("Paired"),
                    linewidth=.5,
                    edgecolor="0",
                    ci=None,
                )
            
            elif analysis_level == 'subject':
                subject_means = plotting_df.groupby(
                    ['method', 'subject'])['r'].mean().reset_index()
                subject_means['task'] = task
                mfx_data.append(subject_means)
                ax = sns.swarmplot(
                    data=subject_means,
                    x="method",
                    y='r',
                    ax=ax,
                    order=method_ordering,
                    alpha=0.75,
                    size=4,
                    edgecolor='gray',
                    linewidth=0.5,
                    palette=sns.color_palette("Paired"),
                )

                # compute mixed effects model
                mfx_results_path = os.path.join(
                    config["mfx_dir"],
                    f'brain-map-correlation-similarities-{task}_mfx-results.csv'
                )
                if not os.path.isfile(mfx_results_path):
                    subject_means = subject_means[subject_means['method'].isin(method_ordering)]
                    subject_means = pd.get_dummies(subject_means, columns=['method'])
                    colname_mapper = {
                        c: c.split('method_')[1]
                        for c in subject_means.columns
                        if 'method' in c
                    }
                    subject_means.rename(columns=colname_mapper, inplace=True)
                    subject_means['DeepLift'] = 0
                    fixed_effects = " + ".join([m for m in method_ordering if m!='DeepLift']) # DeepLift is the reference method
                    model_string = f"r ~ {fixed_effects}"
                    print(
                        f'\nComputing regression model:\n\t{model_string}\n'
                    )
                    mfx_model = bmb.Model(model_string, subject_means)

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
                            f'brain-map-correlation-similarities-{task}_mfx-trace.png'
                        ),
                        dpi=300
                    )
                    mfx_result.to_csv(mfx_results_path)

                else:
                    mfx_result = pd.read_csv(mfx_results_path, index_col=0)

                # plot indicator for meaningful differences
                for mi, method in enumerate(method_ordering):

                    if method != 'DeepLift': # DeepLift is our baseline
                    
                        if 0 < mfx_result.loc[method, 'hdi_3%']:
                            ax.text(
                                s='*',
                                x=mi,
                                y=float(ax.get_ylim()[1])*0.9,
                                ha='center',
                                va='bottom',
                                fontsize=20,
                            )

                ax.set_ylim(-0.5,  float(ax.get_ylim()[1])*1.15)

            else:
                raise ValueError(
                    f'Unknown analysis level: {analysis_level}'
                )
            
            ax.set_xlabel('')
            if analysis_level == 'group':
                ylabel = 'r('+r'$Attr_{Group}; BOLD_{Group}$'+')'
                ax.set_ylim(-.2, None)
            
            elif analysis_level == 'subject':
                ylabel = 'r('+r'$Attr_{Ind}; BOLD_{Group}$'+')'
            
            else:
                raise ValueError(
                    f'Unknown analysis level: {analysis_level}'
                )
            
            if ai == 0:
                ylabel = f'{task}\n\n{ylabel}'

            ax.set_ylabel(ylabel)
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
            'Sfig_brain-maps-correlation-similarity.png'
        ),
        dpi=300
    )

    # compute mixed effects model
    mfx_results_path = os.path.join(
        config["mfx_dir"],
        'brain-map-correlation-similarities_mfx-results.csv'
    )
    if not os.path.isfile(mfx_results_path):
        mfx_data = pd.concat(mfx_data)
        mfx_data = mfx_data[mfx_data['method'].isin(method_ordering)]
        mfx_data = pd.get_dummies(mfx_data, columns=['method'])
        colname_mapper = {
            c: c.split('method_')[1]
            for c in mfx_data.columns
            if 'method' in c
        }
        mfx_data.rename(columns=colname_mapper, inplace=True)
        mfx_data['DeepLift'] = 0
        mfx_effects = [m for m in method_ordering if m!='DeepLift'] # DeepLift is the reference method
        fixed_effects = " + ".join(mfx_effects)
        model_string = f"r ~ {fixed_effects} + ({fixed_effects}|task)"
        print(
            f'\nComputing mixed effects model:\n\t{model_string}\n'
        )
        mfx_model = bmb.Model(model_string, mfx_data)
        
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
                'brain-map-correlation-similarities_mfx-trace.png'
            ),
            dpi=300
        )
        mfx_result.to_csv(mfx_results_path)
    
    else:
        mfx_result = pd.read_csv(mfx_results_path)

    print(mfx_result)

    return fig


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
        '--mfx-dir',
        metavar='DIR',
        default='results/mfx',
        type=str,
        required=False,
        help='path where results of mixed effects analysis are stored '
             '(default: results/mfx)'
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
    parser.add_argument(
        "--seed",
        type=str,
        metavar='INT',
        default=12345,
        help='random seed'
    )
    return parser


if __name__ == '__main__':
    sfig_brain_maps_similarity();