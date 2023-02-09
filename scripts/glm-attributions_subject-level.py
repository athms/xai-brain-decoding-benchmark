#!/usr/bin/env python3

import sys, os
import argparse
import numpy as np
import pandas as pd
from nilearn.image import load_img
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
from nilearn.masking import compute_brain_mask, intersect_masks
import matplotlib.pyplot as plt
sys.path.append('./')
from src import target_labeling
from src.plotting import plot_brain_img, clear_matplotlib_fig_cache


def compute_subject_level_attribution_glm_maps(config=None) -> None:
    """Script's main function; computes subject-level GLM maps
    for attribution data (as resulting from calling scripts/attribute.py)
    """
    
    if config is None:
        config = vars(get_argparse().parse_args())
        config['plot_stat_maps'] = config['plot_stat_maps'] == 'True'

    np.random.seed(config['seed'])

    attribution_methods = [
        p for p in os.listdir(config['attributions_dir'])
        if os.path.isdir(os.path.join(config['attributions_dir'], p))
    ]

    for attribution_method in attribution_methods:
        print(
            '\nComputing subject-level contrasts for '
            f'{attribution_method} attributions..'
        )
        method_attributions_path = os.path.join(
            config['attributions_dir'],
            attribution_method
        )
        glm_data = gather_glm_data(method_attributions_path)
        
        for subject in np.sort(glm_data['subject'].unique()):
            subject_contrasts_path = os.path.join(
                config['subject_level_maps_dir'],
                attribution_method,
                f'sub_{subject}'
            )
            os.makedirs(subject_contrasts_path, exist_ok=True)
            subject_data = glm_data[glm_data['subject'].values == subject].copy()
            design_matrix = make_subject_level_design_matrix(subject_data)
            design_matrix.to_csv(
                os.path.join(
                    subject_contrasts_path,
                    'design_matrix.csv'
                ),
                index=False
            )
            ax = plot_design_matrix(design_matrix=design_matrix)
            ax.set_ylabel('maps')
            plt.savefig(
                os.path.join(
                    subject_contrasts_path,
                    'design_matrix.png'
                ),
                dpi=100
            )
            subject_nii_images = [
                load_img(i)
                for i in subject_data['attribution_image'].values
            ]
            subject_nii_masks = [
                compute_brain_mask(i)
                for i in subject_nii_images
            ]
            subject_mask_img = intersect_masks(subject_nii_masks)
            model = SecondLevelModel(
                smoothing_fwhm=None,
                mask_img=subject_mask_img
            )
            model.fit(subject_nii_images, design_matrix=design_matrix)
            
            for label in target_labeling[config['task']]:
                contrast_string = f'{label}'
                other_label_weight = 1. / (len(target_labeling[config['task']])-1)
                
                for other_label in target_labeling[config['task']]:

                    if other_label != label:
                        contrast_string += f' - {other_label_weight}*{other_label}'
                
                plot_contrast_matrix(
                    contrast_def=contrast_string,
                    design_matrix=design_matrix,
                    output_file=os.path.join(
                        subject_contrasts_path,
                        f'{label}_contrast.png'
                    )
                )
                stat_map = model.compute_contrast(contrast_string)
                stat_map.to_filename(
                    os.path.join(
                        subject_contrasts_path,
                        f'{label}_zmap.nii.gz'
                    )
                )

                if config['plot_stat_maps']:
                    plot_brain_img(
                        img=stat_map,
                        path=os.path.join(
                            subject_contrasts_path,
                            f'{label}_zmap.png'
                        ),
                        workdir=os.path.join(
                            subject_contrasts_path,
                            'tmp/'
                        ),
                        dpi=100
                    )
            
                clear_matplotlib_fig_cache()
            
        print('Done!')

    return None


def gather_glm_data(method_attributions_dir) -> pd.DataFrame:
    """
    Gather data needed to compute the subject-level GLM.

    Parameters
    ----------
    method_attributions_dir : str
        Path to the directory containing the attributions of one
        interpretation method.
    
    Returns
    -------
    data : pd.DataFrame
        Dataframe with the following columns:
        - subject : str
        - attribution_image : str
        - attribution_sum : float
        - label : str
        - fitting_run : str
        - experiment_run : str
    """
    data = []
    fitting_runs = {
        p.split('run-')[1]
        for p in os.listdir(method_attributions_dir)
        if p.startswith('run-')
    }

    for fitting_run in fitting_runs:
        fitting_run_attributions_path = os.path.join(
            method_attributions_dir,
            f'run-{fitting_run}'
        )
        subjects = {
            p.split('sub_')[1]
            for p in os.listdir(fitting_run_attributions_path)
            if p.startswith('sub_')
        }

        for subject in subjects:
            subject_attributions_path = os.path.join(
                fitting_run_attributions_path,
                f'sub_{subject}'
            )
            subject_attributions = [
                os.path.join(subject_attributions_path, p)
                for p in os.listdir(subject_attributions_path)
                if p.endswith('.nii.gz')
            ]
            df_tmp = pd.DataFrame(
                {
                    'subject': subject,
                    'attribution_image': subject_attributions,
                    'attribution_sum': [
                        load_img(a).get_fdata().sum()
                        for a in subject_attributions
                    ],
                    'label': [
                        p.split('/')[-1].split('_')[0]
                        for p in subject_attributions
                    ],
                    'fitting_run': fitting_run,
                }
            )

            if any('run' in p.split('/')[-1] for p in subject_attributions):
                df_tmp['experiment_run'] = [
                    p.split('/')[-1].split('run')[1].split('_')[0]
                    for p in subject_attributions
                ]

            data.append(df_tmp)
    
    return pd.concat(data)


def make_subject_level_design_matrix(subject_data) -> pd.DataFrame:
    """
    Make a design matrix for the subject-level GLM.

    Parameters
    ----------
    subject_data : pd.DataFrame
        Dataframe with the following columns (as generated by gather_glm_data):
        - subject : str
        - attribution_image : str
        - attribution_sum : float
        - label : str
        - fitting_run : str
        - experiment_run : str
    
    Returns
    -------
    design_matrix : pd.DataFrame
    """
    n = subject_data.shape[0]
    design_matrix = {}
    
    for label in subject_data['label'].unique():
        regressor = np.zeros(n)
        regressor[subject_data['label'].values==label] = 1
        design_matrix[label] = regressor

    for fitting_run in subject_data['fitting_run'].unique():
        regressor = np.zeros(n)
        regressor[subject_data['fitting_run'].values==fitting_run] = 1
        design_matrix[f'fit-run-{fitting_run}'] = regressor
    
    if 'experiment_run' in subject_data.columns:
        
        for experiment_run in subject_data['experiment_run'].unique():
            regressor = np.zeros(n)
            regressor[subject_data['experiment_run'].values==experiment_run] = 1
            design_matrix[f'exp-run-{experiment_run}'] = regressor

    design_matrix['attribution_sum'] = subject_data['attribution_sum'].values

    return pd.DataFrame(design_matrix)


def get_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='compute subject-level attribution GLM maps'
    )
    parser.add_argument(
        '--task',
        metavar='DIR',
        default='WM',
        type=str,
        required=False,
        help='name of task; one of [WM, MOTOR, heat-rejection] '
             '(default: WM)'
    )
    parser.add_argument(
        '--attributions-dir',
        metavar='DIR',
        default='results/attributions/task-WM',
        type=str,
        required=False,
        help='directory where attribition images stored '
             '(default: results/attributions/task-WM)'
    )
    parser.add_argument(
        '--subject-level-maps-dir',
        metavar='DIR',
        default='results/glm/attributions/task-WM/subject',
        type=str,
        required=False,
        help='directory where subject-level GLM maps are stored '
             '(default: results/glm/attributions/task-WM/subject)'
    )
    parser.add_argument(
        '--plot-stat-maps',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        required=False,
        help='whether or not to plot individual subject-level maps '
             '(default: False)'
    )
    parser.add_argument(
        '--seed',
        metavar='INT',
        default=12345,
        type=int,
        required=False,
        help='random seed (default: 12345)'
    )

    return parser


if __name__ == '__main__':
    
    compute_subject_level_attribution_glm_maps()