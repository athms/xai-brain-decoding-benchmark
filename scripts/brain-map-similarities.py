#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from nilearn.masking import compute_background_mask, apply_mask
from nilearn.image import resample_to_img


def compute_brain_map_similarities(config=None) -> None:
    """Script's main function; computes similarity of attribution GLM maps
    with BOLD GLM maps and meta analysis maps."""
    
    if config is None:
        config = vars(get_argparse().parse_args())

    np.random.seed(config['seed'])
    random_state = np.random.RandomState(config['seed'])
    
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
            f'{analysis_level}',
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
                        
                        meta_image = f"{image.split('_')[0]}_meta_map.nii.gz"
                        if image.startswith('lh') or image.startswith('rh'):
                            meta_image = 'h_meta_map.nii.gz'
                        if image.startswith('lf') or image.startswith('rf'):
                            meta_image = 'f_meta_map.nii.gz'

                        yield (
                            os.path.join(
                                config["bold_glm_maps_dir"],
                                'group',
                                image
                            ),
                            os.path.join(
                                config["meta_maps_dir"],
                                meta_image
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
                            
                            meta_image = f"{image.split('_')[0]}_meta_map.nii.gz"
                            if image.startswith('lh') or image.startswith('rh'):
                                meta_image = 'h_meta_map.nii.gz'
                            if image.startswith('lf') or image.startswith('rf'):
                                meta_image = 'f_meta_map.nii.gz'
                            
                            yield (
                                os.path.join(
                                    config["bold_glm_maps_dir"],
                                    'group',
                                    image
                                ),
                                os.path.join(
                                    config["meta_maps_dir"],
                                    meta_image
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

            for bold_image_path, meta_image_path, attribution_image_path in yield_image_paths():
                
                (
                    (mi_bold, mi_meta),
                    (r_bold, r_meta)
                ) = compute_similarity_metrics(
                    attribution_image=attribution_image_path,
                    bold_image=bold_image_path,
                    meta_image=meta_image_path,
                    random_state=random_state
                )

                brain_map_similarities.append(
                    pd.DataFrame(
                        data={
                            'method': attribution_method,
                            'mi_bold': mi_bold,
                            'r_bold': r_bold,
                            'mi_meta': mi_meta,
                            'r_meta': r_meta,
                            'bold_image': bold_image_path,
                            'attribution_image': attribution_image_path,
                            'meta_image': meta_image_path,
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


def compute_similarity_metrics(attribution_image, bold_image, meta_image, random_state):
    """helper function to compute mutual inforiation and pearson correlation,
    between attribution image and {bold, meta} images
    
    Args:
    ---
    {attribution,bold,meta}_image : str or nii image
        images or path to images for attribution, bold, and meta-analysis
        images
    
    Returns:
    ---
    (mi_bold, mi_meta), (r_bold, r_meta)
    """
    # reduce data to brain
    mask_img = compute_background_mask(bold_image)
    attribution_masked = apply_mask(attribution_image, mask_img)
    bold_masked = apply_mask(bold_image, mask_img)
    # resample meta analysis image to subject bold
    meta_resampled = resample_to_img(meta_image, bold_image)
    meta_masked = apply_mask(meta_resampled, mask_img)
    # mutual information
    mi_bold = mutual_info_regression(
        X=attribution_masked.reshape(-1,1),
        y=bold_masked,
        discrete_features=False,
        random_state=random_state
    )
    mi_meta = mutual_info_regression(
        X=attribution_masked.reshape(-1,1),
        y=meta_masked,
        discrete_features=False,
        random_state=random_state
    )
    # pearson correlation
    r_bold, _ = pearsonr(attribution_masked, bold_masked)
    r_meta, _ = pearsonr(attribution_masked, meta_masked)
    return (mi_bold, mi_meta), (r_bold, r_meta)


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
        '--meta-maps-dir',
        metavar='DIR',
        default='results/meta_analysis/task-WM',
        type=str,
        required=False,
        help='directory where meta analysis maps are stored '
             '(default: results/meta_analysis/task-WM)'
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
    compute_brain_map_similarities()