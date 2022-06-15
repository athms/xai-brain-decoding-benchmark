#!/usr/bin/env python3

import sys, os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
from nilearn.image import load_img
from nilearn.masking import compute_brain_mask, intersect_masks
sys.path.append('./')
from src import target_labeling
from src.plotting import plot_brain_img, clear_matplotlib_fig_cache


def compute_group_level_attribution_glm_maps(config=None) -> None:
    """Script's main function; computes group-level GLM maps
    for attribution data."""
    
    if config is None:
        config = vars(get_argsparse().parse_args())

    attribution_methods = {
        p for p in os.listdir(config['subject_level_maps_dir'])
        if os.path.isdir(os.path.join(config['subject_level_maps_dir'], p))
    }

    for attribution_method in attribution_methods:
        print(
            '\nComputing group-level contrasts for '
            f'{attribution_method} attributions..'
        )
        subject_level_attributions_path = os.path.join(
            config['subject_level_maps_dir'],
            attribution_method
        )
        glm_data = gather_glm_data(subject_level_attributions_path)
        glm_data = glm_data[glm_data['label'].isin(target_labeling[config['task']])].copy()
        os.makedirs(
            os.path.join(
                config['group_level_maps_dir'],
                attribution_method
            ),
            exist_ok=True
        )
        design_matrix = make_group_level_design_matrix(glm_data)
        design_matrix.to_csv(
            os.path.join(
                config['group_level_maps_dir'],
                attribution_method,
                'design_matrix.csv'
            ),
            index=False
        )
        ax = plot_design_matrix(design_matrix=design_matrix)
        ax.set_ylabel('maps')
        plt.savefig(
            os.path.join(
                config['group_level_maps_dir'],
                attribution_method,
                'design_matrix.png'
            ),
            dpi=100
        )
        nii_images = [load_img(i) for i in glm_data['image'].values]
        nii_masks = [compute_brain_mask(i) for i in nii_images]
        mask_img = intersect_masks(nii_masks)
        model = SecondLevelModel(
            smoothing_fwhm=5.0,
            mask_img=mask_img
        )
        model.fit(nii_images, design_matrix=design_matrix)
        
        for label in target_labeling[config['task']]:
            plot_contrast_matrix(
                contrast_def=label,
                design_matrix=design_matrix,
                output_file=os.path.join(
                    config['group_level_maps_dir'],
                    attribution_method,
                    f'{label}_contrast.png'
                )
            )
            stat_map = model.compute_contrast(label)
            stat_map.to_filename(
                os.path.join(
                    config['group_level_maps_dir'],
                    attribution_method,
                    f'{label}_zmap.nii.gz'
                )
            )
            plot_brain_img(
                img=stat_map,
                path=os.path.join(
                    config['group_level_maps_dir'],
                    attribution_method,
                    f'{label}_zmap.png'
                ),
                workdir=os.path.join(
                    config['group_level_maps_dir'],
                    attribution_method,
                    'tmp/'
                ),
                dpi=300
            )
            clear_matplotlib_fig_cache()

    return None


def gather_glm_data(subject_level_maps_dir) -> pd.DataFrame:
    """
    Gather data needed to compute the group-level GLM.

    Parameters
    ----------
    subject_level_maps_dir : str
        Path to the directory containing the subject-level BOLD GLM maps.
    
    Returns
    -------
    data : pd.DataFrame
        Dataframe with the following columns:
        - subject : str
        - image : str
        - label : str
    """
    data = []
    subjects = {
        p.split('sub_')[1]
        for p in os.listdir(subject_level_maps_dir)
        if p.startswith('sub_')
    }

    for subject in subjects:
        subject_maps_path = os.path.join(
            subject_level_maps_dir,
            f'sub_{subject}'
        )
        subject_maps = [
            os.path.join(subject_maps_path, p)
            for p in os.listdir(subject_maps_path)
            if p.endswith('zmap.nii.gz')
        ]
        df_tmp = pd.DataFrame(
            {
                'subject': subject,
                'image': subject_maps,
                'label': [
                    p.split('/')[-1].split('_')[0]
                    for p in subject_maps
                ]
            }
        )

        data.append(df_tmp)
    
    return pd.concat(data)


def make_group_level_design_matrix(group_data) -> pd.DataFrame:
    """
    Make a design matrix for the group-level GLM.

    Parameters
    ----------
    group_data : pd.DataFrame
        Dataframe with the following columns (as generated by gather_glm_data):
        - subject : str
        - label : str
    
    Returns
    -------
    design_matrix : pd.DataFrame
    """
    n = group_data.shape[0]
    design_matrix = {}
    
    for label in group_data['label'].unique():
        regressor = np.zeros(n)
        regressor[group_data['label'].values==label] = 1
        design_matrix[label] = regressor

    for subject in group_data['subject'].unique():
        regressor = np.zeros(n)
        regressor[group_data['subject'].values==subject] = 1
        design_matrix[f'sub-{subject}'] = regressor
    
    return pd.DataFrame(design_matrix)


def get_argsparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='compute group-level attribution GLM maps.'
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
        '--subject-level-maps-dir',
        metavar='DIR',
        default='results/glm/attributions/task-WM/subject_level',
        type=str,
        required=False,
        help='directory where subject-level GLM maps are stored '
             '(default: results/glm/attributions/task-WM/subject_level)'
    )
    parser.add_argument(
        '--group-level-maps-dir',
        metavar='DIR',
        default='results/glm/attributions/task-WM/group_level',
        type=str,
        required=False,
        help='directory where group-level GLM maps are stored '
             '(default: results/glm/attributions/task-WM/group_level)'
    )

    return parser


if __name__ == '__main__':
    
    compute_group_level_attribution_glm_maps()