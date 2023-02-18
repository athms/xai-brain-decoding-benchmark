#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from nilearn.masking import compute_background_mask, apply_mask
from nilearn.image import mean_img


def sanity_checks_analysis(config=None) -> None:
    """Script's main function; computes sanity check analysis,
    by computing similarity of original attribution brain maps
    with attribution brain maps for models trained on a dataset
    with randomized labels (data randomization test)
    and a model with randomized weights (model randomization test)."""
    
    if config is None:
        config = vars(get_argparse().parse_args())

    np.random.seed(config['seed'])
    random_state = np.random.RandomState(config['seed'])

    os.makedirs(
        os.path.join(
            config["sanity_checks_dir"]
        ),
        exist_ok=True
    )

    attribution_methods = [
        p for p in os.listdir(
            os.path.join(
                config['attributions_dir'],
            )
        )
        if not p.startswith('.')
    ]
    randomized_labels_df = []
    randomized_model_df = []
    df_i = 0

    runs = np.arange(10)

    for run in runs:

        for attribution_method in attribution_methods:

            # randomized labels attributions
            randomized_labels_attributions_dir = os.path.join(
                config['randomized_labels_attributions_dir'],
                attribution_method,
                "run-0"
            )
            randomized_labels_subjects = [
                p.split('sub_')[1]
                for p in os.listdir(randomized_labels_attributions_dir)
                if 'sub' in p
            ]

            # randomized model attributions
            randomized_model_attributions_dir = os.path.join(
                config['randomized_model_attributions_dir'],
                attribution_method,
                "run-0"
            )
            randomized_model_subjects = [
                p.split('sub_')[1]
                for p in os.listdir(randomized_model_attributions_dir)
                if 'sub' in p
            ]
            assert set(randomized_model_subjects)==set(randomized_labels_subjects),\
                f'subjects in {randomized_model_attributions_dir} and {randomized_labels_attributions_dir} do not match'

            # original attributions
            attributions_dir = os.path.join(
                config['attributions_dir'],
                attribution_method,
            )
            run_dir = os.path.join(attributions_dir, f"run-{run}")
            subjects = [
                p.split('sub_')[1]
                for p in os.listdir(run_dir)
                if 'sub' in p
            ]
            assert set(randomized_model_subjects)==set(subjects),\
                f'subjects in {run_dir} do not match with {randomized_model_subjects} and {randomized_labels_subjects}'

            for subject in subjects:
                print(
                    f'Processing: {attribution_method}-attribuitions for sub-{subject} and model fitting run {run}'
                )
                subject_dir = os.path.join(run_dir, f"sub_{subject}")
                attribution_imgs = [
                    p for p in os.listdir(subject_dir)
                    if p.endswith('.nii.gz')
                ]
                # load trial image to compute brain mask
                trial_imgs_dir = os.path.join(
                    config['data_dir'],
                    f"sub_{subject}"
                )
                trial_imgs = [
                    os.path.join(trial_imgs_dir, p)
                    for p in os.listdir(trial_imgs_dir)
                    if '.nii' in p
                ]
                mask_img = compute_background_mask(mean_img(trial_imgs))

                for img in attribution_imgs:
                    randomized_labels_img = os.path.join(randomized_labels_attributions_dir, f"sub_{subject}", img)
                    randomized_labels_img_dat = apply_mask(randomized_labels_img, mask_img).ravel()
                    randomized_model_img = os.path.join(randomized_model_attributions_dir, f"sub_{subject}", img)
                    randomized_model_img_dat = apply_mask(randomized_model_img, mask_img).ravel()
                    img =  os.path.join(subject_dir, img)
                    img_dat = apply_mask(img, mask_img).ravel()
                    rl_mi = mutual_info_regression(
                        X=randomized_labels_img_dat.reshape(-1,1),
                        y=img_dat,
                        discrete_features=False,
                        random_state=random_state
                    )
                    rm_mi = mutual_info_regression(
                        X=randomized_model_img_dat.reshape(-1,1),
                        y=img_dat,
                        discrete_features=False,
                        random_state=random_state
                    )
                    randomized_labels_df.append(
                        pd.DataFrame(
                            data={
                                'method': attribution_method,
                                'mi': rl_mi,
                                'fitting_run': run,
                                'subject': subject,
                                'image': img.split('/')[-1],
                                'contrast': img.split('/')[-1].split('_')[0],
                            },
                            index=[df_i]   
                        )
                    )
                    pd.concat(randomized_labels_df).to_csv(
                        os.path.join(
                            config["sanity_checks_dir"],
                            'sanity-check_randomized-labels.csv'
                        ),
                        index=False
                    )
                    randomized_model_df.append(
                        pd.DataFrame(
                            data={
                                'method': attribution_method,
                                'mi': rm_mi,
                                'fitting_run': run,
                                'subject': subject,
                                'image': img.split('/')[-1],
                                'contrast': img.split('/')[-1].split('_')[0],
                            },
                            index=[df_i]   
                        )
                    )
                    pd.concat(randomized_model_df).to_csv(
                        os.path.join(
                            config["sanity_checks_dir"],
                            'sanity-check_randomized-model.csv'
                        ),
                        index=False
                    )
                    df_i += 1


def get_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='compute sanity checks analysis.'
    )
    parser.add_argument(
        '--task',
        metavar='STR',
        default='WM',
        type=str,
        required=False,
        help='task name; one of [WM, MOTOR, heat-rejection]'
             '(default: WM)'
    )
    parser.add_argument(
        '--data-dir',
        metavar='STR',
        default='data/task-WM',
        type=str,
        required=False,
        help='directory where trial-level GLM maps are stored '
             '(default: data/task-WM)'
    )
    parser.add_argument(
        '--attributions-dir',
        metavar='DIR',
        default='results/attributions/task-WM',
        type=str,
        required=False,
        help='path where attributions for task are stored '
             '(default: results/attributions/task-WM)'
    )
    parser.add_argument(
        '--randomized-labels-attributions-dir',
        metavar='DIR',
        default='results/attributions/randomized_labels/task-WM',
        type=str,
        required=False,
        help='path where attributions for task with randomized labels are stored '
             '(default: results/attributions/randomized_labels/task-WM)'
    )
    parser.add_argument(
        '--randomized-model-attributions-dir',
        metavar='DIR',
        default='results/attributions/randomized_model/task-WM',
        type=str,
        required=False,
        help='path where attributions for task with randomized model are stored '
             '(default: results/attributions/randomized_model/task-WM)'
    )
    parser.add_argument(
        '--sanity-checks-dir',
        metavar='DIR',
        default='results/sanity_checks/task-WM',
        type=str,
        required=False,
        help='path where data of sanity checks analysis are stored '
             '(default: results/sanity_checks/task-WM)'
    )
    parser.add_argument(
        '--seed',
        metavar='INT',
        default=12345,
        type=int,
        required=False,
        help='random seed'
             '(default: 1234)'
    )
    return parser


if __name__ == '__main__':
    sanity_checks_analysis()