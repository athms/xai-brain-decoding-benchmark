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


def compute_subject_level_BOLD_glm_maps(config=None) -> None:
    """Script's main function; computes subject-level BOLD GLM maps."""
    
    if config is None:
        config = vars(get_argparse().parse_args())
        config['plot_stat_maps'] = config['plot_stat_maps'] == 'True'

    task = config['task']
    print(
        f'\nprocessing task: {task}'
    )

    glm_data = gather_glm_data(config['data_dir'])
    
    for subject in glm_data['subject'].unique():
        print(
            f'\tprocessing sub-{subject}'
        )
        subject_contrasts_path = os.path.join(
            config['subject_level_maps_dir'],
            f'sub_{subject}'
        )
        os.makedirs(subject_contrasts_path, exist_ok=True)
        subject_data = glm_data[glm_data['subject'] == subject].copy()
        subject_data = subject_data[subject_data['label'].isin(target_labeling[config['task']])].copy()
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
        subject_nii_images = [load_img(i) for i in subject_data['image'].values]
        subject_nii_masks = [compute_brain_mask(i) for i in subject_nii_images]
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
            
    return None


def gather_glm_data(trial_level_maps_dir) -> pd.DataFrame:
    """
    Gather data needed to compute the subject-level GLM.

    Parameters
    ----------
    trial_level_maps_dir : str
        Path to the directory containing the trial-level BOLD GLM maps.
    
    Returns
    -------
    data : pd.DataFrame
        Dataframe with the following columns:
        - subject : str
        - image : str
        - label : str
        - run : str
    """
    data = []
    subjects = {
        p.split('sub_')[1]
        for p in os.listdir(trial_level_maps_dir)
        if p.startswith('sub_')
    }

    for subject in subjects:
        subject_images_path = os.path.join(
            trial_level_maps_dir,
            f'sub_{subject}'
        )
        subject_images = [
            os.path.join(subject_images_path, p)
            for p in os.listdir(subject_images_path)
            if '.nii' in p
        ]
        df_tmp = pd.DataFrame(
            {
                'subject': subject,
                'image': subject_images,
                'label': [
                    p.split('/')[-1].split('_')[0]
                    for p in subject_images
                ]
            }
        )

        if any('run' in p.split('/')[-1] for p in subject_images):
            df_tmp['run'] = [
                p.split('/')[-1].split('run')[1].split('_')[0]
                for p in subject_images
            ]
        
        data.append(df_tmp)
    
    return pd.concat(data)


def make_subject_level_design_matrix(subject_data: pd.DataFrame) -> pd.DataFrame:
    """
    Make a design matrix for the subject-level GLM.

    Parameters
    ----------
    subject_data : pd.DataFrame
        Dataframe with the following columns (as generated by gather_glm_data):
        - subject : str
        - image : str
        - label : str
        - run : str
    
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

    if 'run' in subject_data.columns:
        for run in subject_data['run'].unique():
            regressor = np.zeros(n)
            regressor[subject_data['run'].values==run] = 1
            design_matrix[f'run-{run}'] = regressor
    
    return pd.DataFrame(design_matrix)


def get_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='compute subject-level BOLD GLM maps'
    )
    parser.add_argument(
        '--task',
        metavar='STR',
        default='WM',
        type=str,
        required=False,
        help='name of task for which to compute subject-level BOLD GLM maps'
             'one of heat-rejection, WM, MOTOR (default: WM)'
    )
    parser.add_argument(
        '--data-dir',
        metavar='DIR',
        default='data/task-WM/trial_images/',
        type=str,
        required=False,
        help='path where trial-level BOLD GLM maps are stored'
             '(default: data/task-WM/trial_images/)'
    )
    parser.add_argument(
        '--subject-level-maps-dir',
        metavar='DIR',
        default='results/glm/BOLD/task-WM/subject_level',
        type=str,
        required=False,
        help='path where subject-level BOLD GLM maps are stored '
             '(default: results/glm/BOLD/task-WM/subject_level)'
    )
    parser.add_argument(
        '--plot-stat-maps',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        required=False,
        help='whether or not to plot subject-level BOLD GLM maps'
             '(default: False)'
    )

    return parser


if __name__ == '__main__':
    
    compute_subject_level_BOLD_glm_maps()