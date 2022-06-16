#!/usr/bin/env python3

import sys, os
import argparse
import json
from typing import Dict
import numpy as np
import pandas as pd
import nilearn as nl
from nilearn.image import load_img, concat_imgs
from nilearn.masking import apply_mask
import collections
import torch
sys.path.append('./')
from src import target_labeling
from src.data import get_subject_trial_image_paths
from src.model import CNNModel


def faithfulness_analysis(config: Dict=None) -> None:
    """Script's main function; computes faithfulness analysis
    for attributions / fitted model of given task."""

    np.random.seed(1234)

    if config is None:
        config = vars(get_argparse().parse_args())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    assert config['task'] in config['fitted_model_dir'], \
        f'model in {config["fitted_model_dir"]} does not match specified task {config["task"]}'
    
    os.makedirs(
        config["faithfulness_dir"],
        exist_ok=True
    )
    # file to store faithfulness analysis results
    faithfulness_path = os.path.join(
        config["faithfulness_dir"],
        'faithfulness_analysis.csv'
    )
    # file that stores occlusion rate at which 
    # decoding performance reaches chance level first
    chance_first_reached_path = os.path.join(
        config["faithfulness_dir"],
        'chance_first_reached_analysis.csv'
    )
    attribution_methods = [
        p for p in os.listdir(config["attributions_dir"])
        if os.path.isdir(os.path.join(config["attributions_dir"], p))
    ]

    with open(
        os.path.join(
            config["fitted_model_dir"],
            'trial_image_paths.json'
            ),
        'r'
        ) as f:
        trial_image_paths = json.load(f)

    fitting_runs = np.sort(
        a=[
            int(d.split('-')[1])
            for d in os.listdir(config["fitted_model_dir"])
            if d.startswith('run-')
        ]
    )
    test_subjects = list(trial_image_paths['test'].keys())
    test_image_paths = get_subject_trial_image_paths(
        path=config["data_dir"],
        subjects=test_subjects,
        decoding_targets=target_labeling[config["task"]].keys()
    )
    test_data = {
        'image_path': [],
        'label': [],
        'numeric_label': [],
        'image': [],
        'nii_image': []
    }

    for subject in test_image_paths:
        
        for image_path in test_image_paths[subject]:
            label = image_path.split('/')[-1].split('_')[0]
            nii_image = load_img(image_path)
            image = np.expand_dims(nii_image.get_fdata(), axis=[0, 1])
            test_data['label'].append(label)
            test_data['numeric_label'].append(target_labeling[config["task"]][label])
            test_data['image'].append(image)
            test_data['image_path'].append(image_path)
            test_data['nii_image'].append(nii_image)

    input_shape = image.shape[1:]
    num_labels = len(target_labeling[config["task"]])

    model_config = json.load(open(os.path.join(config['fitted_model_dir'], 'config.json')))
    model = CNNModel(
        input_shape=input_shape,
        num_classes=num_labels,
        num_filters=model_config["num_filters"],
        filter_size=model_config["filter_size"],
        num_hidden_layers=model_config["num_hidden_layers"],
        dropout=model_config["dropout"]
    )

    def occlude_images(images, attribution_images, mask_img, fraction):
        """Occlude images according to attributions, such that
        the specified fraction of image values receiving the highest
        attribution values are replaces by 0.
        
        Parameters
        ----------
        images : nib.Nifti1Image
            Images to occlude.
        attribution_images : np.ndarray
            Attributions for images.
        mask_img : nib.Nifti1Image
            Brain mask.
        fraction : float
            Fraction of image values to occlude.

        Returns
        -------
        occluded_images : nib.Nifti1Image
            Occluded images.
        """
        images_masked = apply_mask(images, mask_img)
        attributions_masked = apply_mask(attribution_images, mask_img)
        occlusion_idx = np.argsort(a=attributions_masked, axis=1)[:,::-1]
        occlusion_idx = occlusion_idx[:,:int(fraction*occlusion_idx.shape[1])]
        occluded_data = np.array(images_masked)
        occluded_data[:,occlusion_idx] = 0
        return nl.masking.unmask(
            X=occluded_data,
            mask_img=mask_img
        )

    # define occlusion fractions:
    fractions = [
        np.arange(0, 2, 0.2),
        np.arange(2, 5, 0.5),
        np.arange(5, 10, 1),
        np.arange(10, 32, 2)
    ]
    fractions = np.concatenate(fractions)
    fractions = np.round(fractions, 1)
    fraction_start = 0
    faithfulness = []
    chance_first_reached = []
    faithfulness_i = 0
    chance_first_reached_i = 0

    for fitting_run in fitting_runs:
        print(
            f'\nProcessing data for fitting run-{fitting_run}'
        )
        model_path = os.path.join(
            config["fitted_model_dir"],
            f'run-{fitting_run}',
            'final_model.pt'
        )

        if not torch.cuda.is_available():
            model.load_state_dict(
                torch.load(
                    model_path,
                    map_location=torch.device('cpu')
                )
            )

        else:
            model.load_state_dict(torch.load(model_path))
        
        model.eval()

        if torch.cuda.is_available():
            model.to(device)

        test_data['prediction'] = []
        
        for image in test_data['image']:
            image = torch.from_numpy(image).float()
            
            if torch.cuda.is_available():
                image = image.to(device)

            test_data['prediction'].append(
                model(image).detach().cpu().numpy().argmax(axis=1)[0]
        )

        test_data["correct_predictions"] = np.array(test_data['prediction']) == \
            np.array(test_data['numeric_label'])
        
        for k in test_data:
            test_data[k] = list(
                np.array(
                    test_data[k]
                )[test_data["correct_predictions"]]
            )
        
        test_masker = nl.input_data.NiftiMasker(mask_strategy='background')
        fitted_masker = test_masker.fit(imgs=test_data['nii_image'])
        test_mask = fitted_masker.mask_img_
        test_images = concat_imgs(test_data['nii_image'])
        chance_acc = max(collections.Counter(test_data['numeric_label']).values())
        chance_acc /= len(test_data['numeric_label'])
        chance_acc *= 100
        
        for attribution_method in attribution_methods:
            print(
                f'...and {attribution_method} attributions.'
            )
            attribution_images = []

            for test_nii_image_path in test_data['image_path']:
                attribution_paths = get_attribution_paths_for_nii_img(
                    test_nii_image_path,
                    os.path.join(
                        config['attributions_dir'],
                        attribution_method
                    )
                )
                attribution_images.append(
                    load_img(attribution_paths[f'fitting-run-{fitting_run}'])
                )
                                
            attribution_images = concat_imgs(attribution_images)
            chance_first_reached.append(
                pd.DataFrame(
                    {
                        'method': attribution_method,
                        'fitting_run': fitting_run,
                        'fraction_occluded': np.max(fractions),
                        'accuracy': None,
                        'chance': chance_acc
                    },
                    index=[chance_first_reached_i]
                )
            )
            chance_reached = False
            
            for fraction in fractions[fractions>=fraction_start]:
                
                print(
                    '\t... when occluding {:.1f}% of the data.'.format(fraction)
                )
                occluded_images = occlude_images(
                    images=test_images,
                    attribution_images=attribution_images,
                    mask_img=test_mask,
                    fraction=fraction/100.
                )
                occluded_images = np.transpose(occluded_images.get_fdata(), axes=[3, 0, 1, 2])
                occluded_images = np.expand_dims(occluded_images, axis=1)
                occluded_images = torch.tensor(occluded_images).to(torch.float)

                if torch.cuda.is_available():
                    occluded_images = occluded_images.to(device)

                with torch.no_grad():
                    occluded_images_pred = model(occluded_images)
                
                occluded_images_pred = occluded_images_pred.detach().cpu().numpy().argmax(axis=1)
                accuracy = np.mean(
                    occluded_images_pred ==
                    test_data['numeric_label']
                ) * 100
                
                if accuracy <= chance_acc and not chance_reached:
                    chance_first_reached[-1]['fraction_occluded'] = fraction
                    chance_first_reached[-1]['accuracy'] = accuracy
                    chance_first_reached_i += 1
                    chance_reached = True

                faithfulness.append(
                    pd.DataFrame(
                        data={
                            'method': attribution_method,
                            'fitting_run': fitting_run,
                            'fraction_occluded': fraction,
                            'accuracy': accuracy,
                            'chance': chance_acc
                        },
                        index=[faithfulness_i]
                    )
                )
                faithfulness_i += 1

                pd.concat(faithfulness).to_csv(
                    faithfulness_path,
                    index=False
                )

            pd.concat(chance_first_reached).to_csv(
                chance_first_reached_path,
                index=False
            )
            
        print('Done!')
    

def get_attribution_paths_for_nii_img(
    nii_image_path: str=None,
    attributions_dir: str=None
    ) -> Dict[str, str]:
    """
    Returns a dictionary of paths to the attributions for 
    a given nii image for each model fitting run.
    
    Parameters
    ----------
    nii_image_path : str
        Path to the nii image.
    attributions_dir : str
        Path to the directory containing the attributions.
    
    Returns
    -------
    Dict[str, str]
        Dictionary of paths to the attributions for the given nii image.
    """
    subject = nii_image_path.split('/')[-2].split('sub_')[-1]
    filepath = nii_image_path.split('/')[-1].split('.')[0]
    label = filepath.split('_')[0]
    run = filepath.split('run')[1].split('_')[0] if 'run' in filepath else None
    trial = filepath.split('trial')[1].split('_')[0]
    fitting_runs = {
        p.split('run-')[1]
        for p in os.listdir(attributions_dir)
        if 'run' in p  
        and os.path.isdir(os.path.join(attributions_dir, p))
    }
    return {
        f'fitting-run-{fitting_run}': os.path.join(
            attributions_dir,
            f'run-{fitting_run}',
            f'sub_{subject}',
            (
                f'{label}_run{run}_trial{trial}_attributions.nii.gz'
                if run is not None
                else f'{label}_trial{trial}_attributions.nii.gz'
            )
        )
        for fitting_run in fitting_runs
    }


def get_argparse() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description='compute faithulness analysis for attributions of given task.'
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
        '--fitted-model-dir',
        metavar='DIR',
        type=str,
        required=True,
        help='directory of model fitting runs for given task'
             '(as resulting from running scripts/train.py)'
    )
    parser.add_argument(
        '--data-dir',
        metavar='DIR',
        default='data/task-WM/trial_images',
        type=str,
        required=False,
        help='directory where trial-level GLM maps are stored '
             '(default: data/task-WM/trial_images)'
    )
    parser.add_argument(
        '--attributions-dir',
        metavar='DIR',
        default='results/attributions/task-WM',
        type=str,
        required=False,
        help='path where attributions for task are stored '
             '(default: results/attributions/)'
    )
    parser.add_argument(
        '--faithfulness-dir',
        metavar='DIR',
        default='results/faithfulness/task-WM',
        type=str,
        required=False,
        help='path where results of faithfulness analysis are stored '
             '(default: results/faithfulness/task-WM)'
    )

    return parser


if __name__ == '__main__':
    
    faithfulness_analysis()