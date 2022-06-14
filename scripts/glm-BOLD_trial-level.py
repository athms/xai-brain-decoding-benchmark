#!/usr/bin/env python3 

import os
import argparse
import numpy as np
import pandas as pd
from nilearn import image, glm, interfaces
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix


def ev_path(
    task: str,
    subject: str,
    run: int,
    source_path: str
    ) -> str:
    """
    Returns the path to the events file for a given task, subject, and run.

    Parameters
    ----------
    task : str
        The task name.
    subject : str
        The subject ID.
    run : str
        The run number.      
    source_path : str
        The path to the source data.

    Returns
    -------
    path to event file : str
    """
    return os.path.join(
        source_path,
        f"sub-{subject}",
        "func",
        f"sub-{subject}_task-{task}_run-{run}_EV.csv"
    )

def preproc_bold_path(
    task: str,
    subject: str,
    run: str,
    deriv_path: str
    ) -> str:
    """
    Returns the path to preprocessed BOLD file for a given task, subject, and run.

    Parameters
    ----------
    task : str
        The task name.
    subject : str
        The subject ID.
    run : str
        The run number.      
    deriv_path : str
        The path to the fmriprep derivatives.

    Returns
    -------
    path to preprocessed BOLD file : str
    """
    return os.path.join(
        deriv_path,
        f"sub-{subject}",
        "func",
        f"sub-{subject}_task-{task}_run-{run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    )

def preproc_brain_mask_path(
    subject: str,
    task: str,
    run: str,
    deriv_path: str
    ) -> str:
    """
    Returns the path to preprocessed brain mask file for a given task, subject, and run.

    Parameters
    ----------
    task : str
        The task name.
    subject : str
        The subject ID.
    run : str
        The run number.      
    deriv_path : str
        The path to the fmriprep derivatives.

    Returns
    -------
    path to preprocessed brain mask file : str
    """
    return os.path.join(
        deriv_path,
        f"sub-{subject}",
        "func",
        f"sub-{subject}_task-{task}_run-{run}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
    )

def make_design_matrix(
    ev,
    t_rs
    ) -> pd.DataFrame:
    """
    Creates a design matrix for a given event file.

    Parameters
    ----------
    ev : str
        The path to the event file.
    t_rs : array-like
        The timepoints in seconds.
    
    Returns
    -------
    design matrix : pd.DataFrame
    """
    dm = pd.DataFrame()
    
    for trial_i, trial in ev.iterrows():
        regressor = np.zeros_like(t_rs)
        regressor_i = np.logical_and(
            t_rs >= trial.onset,
            t_rs < trial.end
        )
        regressor[regressor_i] = 1
        dm[f'{trial.event_type}_trial-{trial_i}'] = regressor
    
    return dm


def compute_trial_level_BOLD_glm_maps(args: argparse.ArgumentParser=None) -> None:
    """Script's main function; Computes trial-level GLM maps for HCP's WM or MOTOR task."""
    
    if args is None:
        args = get_argparse()

    task = args.task
    print(
        f'\nprocessing task: {task}'
    )
    runs = [1, 2]
    t_r = 0.72
    subjects = list(
        {
            s.split('sub-')[1] 
            for s in os.listdir(args.fmriprep_derivs_dir)
            if os.path.isdir(
                os.path.join(
                    args.source_data_dir,
                    s
                )
            )
            and s.startswith('sub-')
            and 'html' not in s
        }
    )

    for subject in subjects:
        print(
            f'\tprocessing sub-{subject}'
        )
        out_path = os.path.join(
                    args.trial_level_maps_dir,
                    f'sub_{subject}'
        )
        os.makedirs(
            out_path,
            exist_ok=True
        )
        
        for run in runs:
            ev = ev_path(
                task=task,
                subject=subject,
                run=run,
                source_path=args.source_data_dir
            )
            
            if not os.path.isfile(ev):
                print(
                    f"\t\t{ev} does not exist"
                )
                continue
            
            preproc_bold = preproc_bold_path(
                task=task,
                subject=subject,
                run=run,
                deriv_path=args.fmriprep_derivs_dir
            )
            
            if not os.path.isfile(preproc_bold):
                print(
                    f"\t\t{preproc_bold} does not exist"
                )
                continue

            preproc_brain_mask = preproc_brain_mask_path(
                subject=subject,
                task=task,
                run=run,
                deriv_path=args.fmriprep_derivs_dir
            )
            
            if not os.path.isfile(preproc_brain_mask):
                preproc_brain_mask = None
            
            confounds, _ = interfaces.fmriprep.load_confounds(
                img_files=preproc_bold,
                strategy=(
                    'motion',
                    'high_pass',
                    'wm_csf',
                    'global_signal'
                ),
                motion='full',
                wm_csf='basic',
                global_signal='basic',
                demean=True
            )
            ev = pd.read_csv(
                ev,
                index_col=False
            )
            ev['trial_type'] = [
                f'{event_type}_run{run}_trial{i}'
                for i, event_type in enumerate(ev['event_type'])
            ]
            preproc_bold = image.load_img(preproc_bold)
            design_matrix = glm.first_level.make_first_level_design_matrix(
                frame_times=np.arange(preproc_bold.shape[-1]) * t_r,
                events=ev[['onset', 'duration', 'trial_type']],
                hrf_model='glover',
                drift_model=None,
                high_pass=None,
                add_regs=confounds,
                min_onset=-24,
                oversampling=50
            )
            design_matrix.to_csv(
                os.path.join(
                    out_path,
                    'design_matrix.csv'
                ),
                index=False
            )
            plot_design_matrix(
                design_matrix=design_matrix,
                output_file=os.path.join(
                    out_path,
                    'design_matrix.png'
                )
            );
            trial_level_model = glm.first_level.FirstLevelModel(
                t_r=t_r,
                slice_time_ref=0.5,
                hrf_model='glover',
                drift_model=None,
                high_pass=None,
                min_onset=-24,
                mask_img=preproc_brain_mask,
                smoothing_fwhm=None,
                standardize=False,
                signal_scaling=0,
                noise_model='ar1',
            )
            trial_level_model = trial_level_model.fit(
                run_imgs=preproc_bold,
                design_matrices=design_matrix
            )
            contrast_matrix = np.eye(design_matrix.shape[1])
            trial_level_contrasts = dict(
                [
                    (
                        trial_type,
                        contrast_matrix[design_matrix.columns==trial_type]
                    )
                    for trial_type in ev['trial_type']
                ]
            )

            for contrast_name, contrast in trial_level_contrasts.items():
                trial_level_contrast_path = os.path.join(
                    out_path,
                    f'{contrast_name}.nii.gz'
                )
                
                if not os.path.isfile(trial_level_contrast_path) or args.overwrite=='True':
                    trial_level_map = trial_level_model.compute_contrast(
                        contrast,
                        output_type='z_score'
                    )
                    trial_level_map.to_filename(trial_level_contrast_path)
                    plot_contrast_matrix(
                        contrast_def=contrast,
                        design_matrix=design_matrix,
                        output_file=os.path.join(
                            out_path,
                            f'{contrast_name}_contrast.png'
                        )
                    )
                
                else:
                    print(
                        f'\t\t{trial_level_contrast_path} exists already.'
                    )


def get_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='compute trial-level BOLD GLM maps for HCP tasks.'
    )
    parser.add_argument(
        "--task",
        type=str,
        metavar='STR',
        required=False,
        default='WM',
        help='task for which to compute trial-level GLM BOLD maps'
             'one of [WM, MOTOR] (default: WM)'
    )
    parser.add_argument(
        "--source-data-dir",
        type=str,
        metavar='DIR',
        required=True,
        help='path to HCP source data'
    )
    parser.add_argument(
        "--fmriprep-derivs-dir",
        type=str,
        metavar='DIR',
        required=True,
        help='path to HCP fmriprep derivatives'
    )
    parser.add_argument(
        "--trial-level-maps-dir",
        type=str,
        metavar='DIR',
        required=False,
        default='data/task-WM/trial_images',
        help='path where trial-level GLM maps are stored'
             '(default: data/task-WM/trial_images)'
    )
    parser.add_argument(
        "--overwrite",
        type=str,
        metavar='BOOL',
        choices=('True', 'False'),
        default='True',
        help='whether or not to overwrite existing trial-level map files'
             'one of [True, False] (default: True)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":

    compute_trial_level_BOLD_glm_maps()