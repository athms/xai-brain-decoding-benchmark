#!/usr/bin/env python3 

import os
import numpy as np
import nibabel as nb
import nilearn
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple


def get_subject_contrast_image_paths(
      path: str='data/contrast_images/',
      contrast_labels: List[str]=None,
      subjects: Tuple[int]=None
    ) -> Dict:
    
    if contrast_labels is None:
        contrast_labels = [
            'friend',
            'heat',
            'rejection',
            'warmth'
        ]
    
    if subjects is None:
        subjects = np.array(
            [
                f'{s:02}'
                for s in range(1, 60)
            ]
        )

    contrast_image_paths = {
        label: []
        for label in contrast_labels
    }
    
    for label in contrast_labels:
        
        for subject in subjects:

            image_path = os.path.join(
                path,
                '{}_sub_{}.nii'.format(
                    label, subject
                )
            )
            
            if os.path.isfile(image_path):
                contrast_image_paths[label].append(image_path)
    
    return contrast_image_paths


def get_subject_trial_image_paths(
      path: str,
      decoding_targets: Tuple[str],
      subjects: Tuple[int],
    ) -> Dict:

    trial_image_paths = {}

    for subject in subjects:

        subject_dir = os.path.join(
            path,
            'sub_{}'.format(subject)
        )

        trial_image_paths[subject] = [
            os.path.join(subject_dir, f)
            for f in os.listdir(subject_dir)
            if np.logical_or(
                f.endswith('.nii'), 
                f.endswith('nii.gz')
            )
            and any(
                target in f
                for target in decoding_targets
            )
        ]
    
    return trial_image_paths


def assign_labels(
    image_paths: Dict[str, str],
    target_labeling: Dict[str, int]
    ) -> Tuple:

    labels = []
    
    for _, paths in image_paths.items():
        
        for path in paths:
            
            for label, cl in target_labeling.items():
                
                if label in path.split('/')[-1]:
                    labels.append(cl)
                    break
    
    return np.array(labels)


def load_images(
    image_paths: Dict[str, str],
    target_labeling: Dict[str, int],
    return_fdata: bool=False,
    smoothing_fwhm: float=None
    ) -> List:

    images = []
    
    for _, paths in image_paths.items():
        
        for path in paths:
            
            for label in target_labeling:
                
                if label in path.split('/')[-1]:

                    images.append(
                        nilearn.image.smooth_img(
                            imgs=nb.load(path),
                            fwhm=smoothing_fwhm
                        )
                    )
                        
                    break
    
    if return_fdata:

        images = np.concatenate(
            [
                np.expand_dims(
                    a=image.get_fdata(),
                    axis=0
                )
                for image in images
            ], 
            axis=0
        )
    
    return images


def load_data(
    image_paths: Dict[str, str],
    target_labeling: Dict[str, int],
    return_fdata: bool=False,
    smoothing_fwhm: float=None
    ) -> Tuple[List, List]:

    images = load_images(
        image_paths=image_paths,
        target_labeling=target_labeling,
        return_fdata=return_fdata,
        smoothing_fwhm=smoothing_fwhm
    )

    labels = assign_labels(
        image_paths=image_paths,
        target_labeling=target_labeling
    )
    
    return (images, labels)