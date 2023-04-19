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


def test_hdi(hdi: np.ndarray, true_value: float = 0) -> bool:
    """Test whether true value is in HDI"""
    return hdi[0] <= true_value <= hdi[1]


def fig_brain_maps_similarity(config=None) -> None:
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
        config["regr_dir"],
        exist_ok=True
    )

    fig, fig_axs = plt.subplot_mosaic(
        """
        ABB
        CDD
        EFF
        """,
        figsize=(8, 6),
        dpi=300,
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
    regr_data = []

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
                'mi': brain_map_similarities[f'mi_bold'],
                'Reference': 'Group BOLD',
                'subject': subjects
            })
            plotting_df = pd.concat([
                plotting_df,
                pd.DataFrame({
                    'method': brain_map_similarities['method'],
                    'mi': brain_map_similarities[f'mi_meta'],
                    'Reference': 'Meta-analysis',
                    'subject': subjects
                    
                })
            ])  

            if analysis_level in {'group'}:
                ax = sns.barplot(
                    data=plotting_df,
                    x="method",
                    y='mi',
                    hue='Reference',
                    ax=ax,
                    order=method_ordering,
                    palette=sns.color_palette("colorblind",2),
                    linewidth=.5,
                    edgecolor="0",
                    ci=None,
                )
                if task != 'heat-rejection':
                    ax.legend_.remove()
            
            elif analysis_level == 'subject':
                subject_means = plotting_df.groupby(
                    ['method', 'Reference', 'subject'])['mi'].mean().reset_index()
                subject_means['task'] = task
                regr_data.append(subject_means)
                ax = sns.violinplot(
                    data=subject_means,
                    x="method",
                    y='mi',
                    hue='Reference',
                    ax=ax,
                    order=method_ordering,
                    alpha=0.75,
                    size=4,
                    edgecolor='gray',
                    linewidth=0.5,
                    palette=sns.color_palette("colorblind",2),
                    scale_hue=True,
                )
                ax.legend_.remove()

                for ri, reference in enumerate(subject_means['Reference'].unique()):
                    # compute mixed effects model
                    regr_results_path = os.path.join(
                        config["regr_dir"],
                        f'brain-map-similarities-{task}_ref-{reference}_regr-results.csv'
                    )
                    if not os.path.isfile(regr_results_path):
                        subject_means_ref = subject_means[subject_means['Reference']==reference].copy()
                        subject_means_ref = subject_means_ref[subject_means_ref['method'].isin(method_ordering)]
                        subject_means_ref = pd.get_dummies(subject_means_ref, columns=['method'])
                        colname_mapper = {
                            c: c.split('method_')[1]
                            for c in subject_means_ref.columns
                            if 'method' in c
                        }
                        subject_means_ref.rename(columns=colname_mapper, inplace=True)
                        subject_means_ref['DeepLift'] = 0
                        fixed_effects = " + ".join([m for m in method_ordering if m!='DeepLift']) # DeepLift is the reference method
                        model_string = f"mi ~ {fixed_effects}"
                        print(
                            f'\nComputing regression model for {reference} reference:\n\t{model_string}\n'
                        )
                        regr_model = bmb.Model(model_string, subject_means_ref)

                        n_tune = 5000
                        converged = False
                        while not converged:
                            results = regr_model.fit(
                                draws=5000,
                                tune=n_tune,
                                chains=4,
                                random_seed=config['seed']
                            )
                            regr_result = az.summary(results)

                            if all(np.abs(regr_result['r_hat']-1) < .05):
                                converged = True
                            
                            n_tune += 5000

                        az.plot_trace(results);
                        plt.tight_layout()
                        plt.savefig(
                            fname=os.path.join(
                                config["regr_dir"],
                                f'brain-map-similarities-{task}_ref-{reference}_regr-trace.png'
                            ),
                            dpi=300
                        )
                        regr_result.to_csv(regr_results_path)

                    else:
                        regr_result = pd.read_csv(regr_results_path, index_col=0)

                    # plot indicator for meaningful differences
                    for mi, method in enumerate(method_ordering):

                        if method != 'DeepLift': # DeepLift is our baseline
                        
                            if 0 < regr_result.loc[method, 'hdi_3%']:
                                ax.text(
                                    s='*',
                                    x=mi-0.2 if ri==0 else mi+0.2,
                                    y=float(ax.get_ylim()[1])*0.9,
                                    ha='center',
                                    va='bottom',
                                    fontsize=20,
                                    color='k' #sns.color_palette("colorblind",2)[ri]
                                )

                ax.set_ylim(0,  float(ax.get_ylim()[1])*1.15)

            else:
                raise ValueError(
                    f'Unknown analysis level: {analysis_level}'
                )
            
            ax.set_xlabel('')
            if analysis_level == 'group':
                ylabel = r'$I(Attr_{Group}; Ref)$'
            
            elif analysis_level == 'subject':
                ylabel = r'$I(Attr_{Ind}; Ref)$'
            
            else:
                raise ValueError(
                    f'Unknown analysis level: {analysis_level}'
                )
            
            if ai == 0:
                ylabel = f'{task}\n\n{ylabel}'

            ax.set_ylabel(ylabel)
            ax.set_ylim(0, None)
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
            'Fig-3_brain-maps-similarity.jpg'
        ),
        dpi=300
    )

    # compute mixed effects model
    regr_results_path = os.path.join(
        config["regr_dir"],
        'brain-map-similarities_regr-results.csv'
    )
    if not os.path.isfile(regr_results_path):
        regr_data = pd.concat(regr_data)
        regr_data = regr_data[regr_data['method'].isin(method_ordering)]
        regr_data = pd.get_dummies(regr_data, columns=['method'])
        colname_mapper = {
            c: c.split('method_')[1]
            for c in regr_data.columns
            if 'method' in c
        }
        regr_data.rename(columns=colname_mapper, inplace=True)
        regr_data['DeepLift'] = 0
        regr_effects = [m for m in method_ordering if m!='DeepLift'] # DeepLift is the reference method
        fixed_effects = " + ".join(regr_effects)
        model_string = f"mi ~ {fixed_effects} + ({fixed_effects}|task)"
        print(
            f'\nComputing mixed effects model:\n\t{model_string}\n'
        )
        regr_model = bmb.Model(model_string, regr_data)
        
        n_tune = 5000
        converged = False
        while not converged:
            results = regr_model.fit(
                draws=5000,
                tune=n_tune,
                chains=4,
                random_seed=config['seed']
            )
            regr_result = az.summary(results)
            
            if all(np.abs(regr_result['r_hat']-1) < .05):
                converged = True
            
            n_tune += 5000

        az.plot_trace(results);
        plt.tight_layout()
        os.makedirs(
            config["regr_dir"],
            exist_ok=True
        )
        plt.savefig(
            fname=os.path.join(
                config["regr_dir"],
                'brain-map-similarities_regr-trace.png'
            ),
            dpi=300
        )
        regr_result.to_csv(regr_results_path)
    
    else:
        regr_result = pd.read_csv(regr_results_path)

    print(regr_result)

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
        '--regr-dir',
        metavar='DIR',
        default='results/regr',
        type=str,
        required=False,
        help='path where results of mixed effects analysis are stored '
             '(default: results/regr)'
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
    fig_brain_maps_similarity();