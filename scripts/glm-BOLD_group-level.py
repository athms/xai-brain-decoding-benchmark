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
from nilearn.glm import threshold_stats_img
sys.path.append('./')
from src import target_labeling
from src.plotting import plot_brain_img, clear_matplotlib_fig_cache


def compute_group_level_BOLD_glm_maps(config=None) -> None:
    """Script's main function; computes group-level BOLD GLM maps."""
    
    if config is None:
        config = vars(get_argparse().parse_args())

    np.random.seed(config['seed'])

    glm_data = gather_glm_data(config['subject_level_maps_dir'])
    glm_data = glm_data[glm_data['label'].isin(target_labeling[config['task']])].copy()
    os.makedirs(config['group_level_maps_dir'], exist_ok=True)
    design_matrix = make_group_level_design_matrix(glm_data)
    design_matrix.to_csv(
        os.path.join(
            config['group_level_maps_dir'],
            'design_matrix.csv'
        ),
        index=False
    )
    ax = plot_design_matrix(design_matrix=design_matrix)
    ax.set_ylabel('maps')
    plt.savefig(
        os.path.join(
            config['group_level_maps_dir'],
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
                f'{label}_contrast.png'
            )
        );
        stat_map = model.compute_contrast(label)
        stat_map.to_filename(
            os.path.join(
                config['group_level_maps_dir'],
                f'{label}_zmap.nii.gz'
            )
        )
        # stat_map_thresholded, threshold = threshold_stats_img(
        #     stat_img=stat_map,
        #     mask_img=mask_img,
        #     alpha=0.01,
        #     height_control='fpr'
        # )
        plot_brain_img(
            img=stat_map,
            path=os.path.join(
                config['group_level_maps_dir'],
                f'{label}_zmap.png'
            ),
            workdir=os.path.join(
                config['group_level_maps_dir'],
                'tmp/'
            ),
            threshold=None,
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


def get_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='compute group-level BOLD GLM maps'
    )
    parser.add_argument(
        '--task',
        metavar='STR',
        default='WM',
        type=str,
        required=False,
        help='name of task for which to compute group-level GLM BOLD maps'
             '(default: WM)'
    )
    parser.add_argument(
        '--subject-level-maps-dir',
        metavar='DIR',
        default='results/glm/BOLD/task-WM/subject',
        type=str,
        required=False,
        help='path where subject-level BOLD GLM maps are stored'
             '(default: results/glm/BOLD/task-WM/subject)'
    )
    parser.add_argument(
        '--group-level-maps-dir',
        metavar='DIR',
        default='results/glm/BOLD/task-WM/group',
        type=str,
        required=False,
        help='path where group-level BOLD GLM maps are stored'
             '(default: results/glm/BOLD/task-WM/group)'
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
    
    compute_group_level_BOLD_glm_maps()