#!/usr/bin/env python3

import os
import argparse
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from nilearn.masking import compute_background_mask, apply_mask


def compute_brain_map_similarities(config=None) -> None:
    """Script's main function; computes similarity of BOLD and attribution 
    GLM maps (as measured by mutual information)."""
    
    if config is None:
        config = vars(get_argparse().parse_args())
    
    os.makedirs(
        os.path.join(
            config["brain_maps_similarity_dir"]
        ),
        exist_ok=True
    )
    print(
        '\nComputing similarity of group-level BOLD GLM maps and ...'
    )
    for analysis_level in ['group', 'subject']:
        print(
            f'... {analysis_level}-level attribution GLM maps'
        )

        attribution_maps_dir = os.path.join(
            config["attribution_glm_maps_dir"],
            f'{analysis_level}_level',
        )
        attribution_methods = [
            p for p in os.listdir(attribution_maps_dir)
            if not p.startswith('.')
        ]
        brain_map_similarities = []
        brain_map_similarities_i = 0

        for attribution_method in attribution_methods:
            print(
                f'\t... for {attribution_method} attributions'
            )

            if analysis_level == 'group':
                
                def yield_image_paths():
                    base_path = os.path.join(
                        attribution_maps_dir,
                        attribution_method
                    )
                    attribution_image_paths = {
                        p for p in os.listdir(base_path)
                        if p.endswith('.nii.gz')
                    }
                    for image in attribution_image_paths:
                        yield (
                            os.path.join(
                                config["bold_glm_maps_dir"],
                                'group_level',
                                image
                            ),
                            os.path.join(
                                base_path,
                                image
                            )
                        )

            elif analysis_level == 'subject':
                
                def yield_image_paths():
                    base_path = os.path.join(
                        attribution_maps_dir,
                        attribution_method
                    )
                    subjects = {
                        p.split('sub_')[1]
                        for p in os.listdir(base_path)
                        if p.startswith('sub_')
                    }
                    for subject in subjects:
                        attribution_image_paths = {
                            p for p in os.listdir(
                                os.path.join(
                                    base_path,
                                    f'sub_{subject}'
                                )
                            )
                            if p.endswith('.nii.gz')
                        }
                        for image in attribution_image_paths:
                            yield (
                                os.path.join(
                                    config["bold_glm_maps_dir"],
                                    'group_level',
                                    image
                                ),
                                os.path.join(
                                    attribution_maps_dir,
                                    attribution_method,
                                    f'sub_{subject}',
                                    image
                                )
                            )

            else:
                raise ValueError(
                    f'Unknown analysis level: {analysis_level}'
                )

            
            for bold_image_path, attribution_image_path in yield_image_paths():
                mask_img = compute_background_mask(bold_image_path)
                mi = mutual_info_regression(
                    X=apply_mask(attribution_image_path, mask_img).reshape(-1,1),
                    y=apply_mask(bold_image_path, mask_img),
                    discrete_features=False
                )
                brain_map_similarities.append(
                    pd.DataFrame(
                        data={
                            'method': attribution_method,
                            'mi': mi,
                            'bold_image': bold_image_path,
                            'attribution_image': attribution_image_path,
                            'contrast': bold_image_path.split('/')[-1].split('_')[0]
                        },
                        index=[brain_map_similarities_i]               
                    )
                )
                brain_map_similarities_i += 1
                pd.concat(brain_map_similarities).to_csv(
                    os.path.join(
                        config["brain_maps_similarity_dir"],
                        f'brain-map-similarities_{analysis_level}.csv'
                    ),
                    index=False
                )

            print('Done!')


def get_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='compute similariy of BOLD and attribution brain maps.'
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
        '--bold-glm-maps-dir',
        metavar='DIR',
        default='results/glm/BOLD/task-WM',
        type=str,
        required=False,
        help='directory where BOLD GLM maps are stored '
             '(default: results/glm/BOLD/task-WM)'
    )
    parser.add_argument(
        '--attribution-glm-maps-dir',
        metavar='DIR',
        default='results/glm/attributions/task-WM',
        type=str,
        required=False,
        help='directory where attribution GLM maps are stored '
             '(default: results/glm/attributions/task-WM)'
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
        '--brain-maps-similarity-dir',
        metavar='DIR',
        default='results/brain_map_similarity/task-WM',
        type=str,
        required=False,
        help='path where data of analysis are stored '
             '(default: results/brain_map_similarity/task-WM)'
    )
    return parser


if __name__ == '__main__':
    
    compute_brain_map_similarities()