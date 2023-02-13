#!/usr/bin/env python3 

import os, sys
import argparse
import json
from typing import Dict
import numpy as np
import pandas as pd
from nilearn.image import load_img, concat_imgs
from nilearn.masking import compute_background_mask, apply_mask
import torch
sys.path.append('./')
from src import target_labeling
from src.data import get_subject_trial_image_paths
from src.model import CNNModel
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(
    context='paper',
    style='ticks'
)


def fig_attribution_variance(config: Dict=None) -> None:
    """Script's main function; computes variance across
    attributions and correlates mean variance with
    decoding accuracy.
    """

    if config is None:
        config =  vars(get_argparse().parse_args())

    os.makedirs(
        config["figures_dir"],
        exist_ok=True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fig, fig_axs = plt.subplots(
        3, 9, # tasks x attribution methods
        figsize=(14, 5),
        dpi=300,
        sharex='row'
    )

    for task_i, task in enumerate(
        [
            'heat-rejection',
            'MOTOR',
            'WM'
        ]
    ):
        print(f'\nProcessing task-{task}...')

        model_dir = [
            os.path.join(config['fitted_models_base_dir'], p)
            for p in os.listdir(config['fitted_models_base_dir'])
            if p.startswith(f"task-{task}")
        ]
        assert len(model_dir) == 1, f"too many models ({len(model_dir)}) found for task {task}"
        model_dir = model_dir[0]
        model_config = json.load(open(os.path.join(model_dir, 'config.json')))

        with open(
            os.path.join(
                config['data_base_dir'],
                f'task-{task}',
                'train_test_split.json'
                ),
            'r'
        ) as f:
            trial_image_paths = json.load(f)

        fitting_runs = np.sort(
            a=[
                int(d.split('-')[1])
                for d in os.listdir(model_dir)
                if d.startswith('run-')
            ]
        )
        test_subjects = list(trial_image_paths['test'].keys())
        test_image_paths = get_subject_trial_image_paths(
            path=os.path.join(config["data_base_dir"], f'task-{task}'),
            subjects=test_subjects,
            decoding_targets=target_labeling[task].keys()
        )

        test_data = {}
        for subject in test_image_paths:
            test_data[subject]= {
                'image_path': [],
                'label': [],
                'numeric_label': [],
                'image': [],
                'nii_image': []
            }
            for image_path in test_image_paths[subject]:
                label = image_path.split('/')[-1].split('_')[0]
                nii_image = load_img(image_path)
                image = np.expand_dims(nii_image.get_fdata(), axis=[0, 1])
                test_data[subject]['label'].append(label)
                test_data[subject]['numeric_label'].append(target_labeling[task][label])
                test_data[subject]['image'].append(image)
                test_data[subject]['image_path'].append(image_path)
                test_data[subject]['nii_image'].append(nii_image)

        num_labels = len(np.unique(test_data[subject]['numeric_label']))
        input_shape = image.shape[1:]

        model = CNNModel(
            input_shape=input_shape,
            num_classes=num_labels,
            num_filters=model_config["num_filters"],
            filter_size=model_config["filter_size"],
            num_hidden_layers=model_config["num_hidden_layers"],
            dropout=model_config["dropout"]
        )

        # compute decoding accuracies:
        df_i, df_acc = 0, []
        for fitting_run in fitting_runs:
            model_path = os.path.join(
                model_dir,
                f'run-{fitting_run}',
                'best_model.pt'
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
            
            for subject in test_data:
                preds = []
                for image in test_data[subject]['image']:
                    image = torch.from_numpy(image).float()
                    if torch.cuda.is_available():
                        image = image.to(device)
                    preds.append(model(image).detach().cpu().numpy().argmax(axis=1)[0])
                df_acc.append(pd.DataFrame({
                    'subject': subject,
                    'acc': np.mean(
                        np.array(preds)==test_data[subject]['numeric_label']
                    ) * 100
                }, index=[df_i]))
                df_i += 1
        df_acc = pd.concat(df_acc)

        # compute attribution variance:
        attribution_methods = [
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
        df_i, df = 0, []
        for subject in test_data:
            print(f'\t...computing attribution variance for sub-{subject}')
            for method in attribution_methods:
                attributions = []
                for fitting_run in fitting_runs:    
                    subject_attributions = os.path.join(
                        config['attributions_base_dir'],
                        f'task-{task}',
                        method,
                        f'run-{fitting_run}',
                        f'sub_{subject}'
                    )
                    images = []
                    for image in np.sort([
                        i for i in os.listdir(subject_attributions)
                        if i.endswith('.nii.gz')
                    ]):
                        images.append(load_img(os.path.join(subject_attributions, image)))
                    images = concat_imgs(images)
                    attributions.append(images)
                assert all([im.shape[-1]==images.shape[-1] for im in attributions])
                attributions_stacked = concat_imgs(attributions)
                mask = compute_background_mask(attributions_stacked)
                attributions = apply_mask(attributions_stacked, mask).reshape(len(attributions), images.shape[-1], -1)
                var = np.mean([np.var(attributions[:,i], axis=0) for i in range(attributions.shape[1])])
                df.append(
                    pd.DataFrame({
                        'subject': subject,
                        'method': method,
                        'var': var,
                        'acc': df_acc[df_acc['subject']==subject]['acc'].values[0]
                    }, index=[df_i]) 
                )
                df_i += 1
        df = pd.concat(df)
        
        for i, method in enumerate(attribution_methods):
            ax = sns.regplot(
                data=df[df['method']==method].copy(),
                x='acc',
                y='var',
                ax=fig_axs[task_i, i]
            )
            if task_i==0:
                ax.set_title(method)
            if i>0:
                ax.set_ylabel('')
            else:
                ax.set_ylabel('Var(Attributions)')
            if task_i<2:
                ax.set_xlabel('')
            else:
                ax.set_xlabel('Accuracy')

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                config['figures_dir'],
                f'Fig-X_attribution-variance.png'
            ),
            dpi=300
        )
        print('..done.')
        

def get_argparse(parser: argparse.ArgumentParser=None) -> argparse.ArgumentParser:
    
    if parser is None:
        parser = argparse.ArgumentParser(
            description='create decoding performance figure',
        )

    parser.add_argument(
        '--attributions-base-dir',
        metavar='DIR',
        type=str,
        default='results/attributions',
        help='directory where attribution data are stored '
             'for each task (default: results/attributions)'
    )
    parser.add_argument(
        '--fitted-models-base-dir',
        metavar='DIR',
        type=str,
        default='results/models',
        help='directory where final model fitting runs are stored '
             'for each task (default: results/models)'
    )
    parser.add_argument(
        '--data-base-dir',
        metavar='DIR',
        type=str,
        default='data',
        help='path to base directory where trial-level GLM maps are stored '
             'for all tasks (default: data/)'
    )
    parser.add_argument(
        '--figures-dir',
        metavar='DIR',
        default='figures',
        type=str,
        help='path where figure is saved (default: figures)'
    )
    
    return parser


if __name__ == '__main__':
    fig_attribution_variance()