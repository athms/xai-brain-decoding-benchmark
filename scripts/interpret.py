#!/usr/bin/env python3

import sys, os
from typing import Dict
import argparse
import json
import numpy as np
import nibabel as nb
import nilearn as nl
import torch
from captum.attr import \
    DeepLift, \
    DeepLiftShap, \
    GuidedBackprop, \
    GuidedGradCam, \
    InputXGradient, \
    IntegratedGradients, \
    NoiseTunnel, \
    Saliency
from zennit.composites import LayerMapComposite
from zennit.rules import Epsilon, Gamma, Pass
from zennit.types import Convolution, Linear, Activation
from zennit.canonizers import SequentialMergeBatchNorm
sys.path.append('./')
from src.data import load_data, get_subject_trial_image_paths
from src.model import CNNModel
from src import target_labeling


def interpret(config: Dict=None) -> None:
    """Script's main function; interprets model decoding decisions
    for given task with studied interpretation methods."""
    
    if config is None:
        config = vars(attribute_argsparse().parse_args())
        config['use_random_init'] = config['use_random_init'] == 'True'

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    assert config['task'] in config['fitted_model_dir'], \
        f'model in {config["fitted_model_dir"]} does not match specified task {config["task"]}'
    model_config = json.load(open(os.path.join(config['fitted_model_dir'], 'config.json')))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # setup LRP attribution
    composite_lrp_map = [
        (Activation, Pass()), 
        (Convolution, Gamma(gamma=0.25)), # according to: https://link-springer-com.stanford.idm.oclc.org/chapter/10.1007/978-3-030-28954-6_10
        (Linear, Epsilon(epsilon=0)) # epsilon=0 -> LRP-0 rule
    ]
    canonizer = SequentialMergeBatchNorm()
    LRP = LayerMapComposite(
        layer_map=composite_lrp_map,
        canonizers=[canonizer]
    )
    LRP.__name__ = 'LRP'

    # rename Saliency -> Gradient
    Gradient = Saliency
    Gradient.__name__ = 'Gradient'

    # setup dummy-class for SmoothGrad attribution
    class SmoothGradDummy:
        
        def __init__(self):
            self.__name__ = 'SmoothGrad'
        
        def __call__(self, model):
            return NoiseTunnel(Saliency(model))
    
    SmoothGrad = SmoothGradDummy()
        
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
            d.split('-')[1]
            for d in os.listdir(config["fitted_model_dir"])
            if d.startswith('run-')
        ]
    )

    if config['use_random_init']:
        fitting_runs = fitting_runs[0]

    test_subjects = list(trial_image_paths['test'].keys())
    test_image_paths =  get_subject_trial_image_paths(
        path=config["data_dir"],
        subjects=test_subjects,
        decoding_targets=target_labeling[config["task"]].keys()
    )
    test_images, _ = load_data(
        image_paths=test_image_paths,
        return_fdata=True,
        target_labeling=target_labeling[config["task"]]
    )
    # adding "channel"-dimension
    test_images = np.expand_dims(
        a=test_images,
        axis=1
    )
    input_shape = test_images.shape[1:]
    num_labels = len(target_labeling[config["task"]])

    for run in fitting_runs:
        print(
            f'\nComputing attributions for training run {run} ...'
        )
        model_path = os.path.join(
            config["fitted_model_dir"],
            f'run-{run}',
            'final_model.pt'
        )
        attribution_methods = [
            DeepLift,
            DeepLiftShap,
            GuidedBackprop,
            GuidedGradCam,
            InputXGradient,
            IntegratedGradients,
            LRP,
            Gradient,
            SmoothGrad
        ]

        for attribution_method in attribution_methods:
            name_attribution_method = str(attribution_method.__name__)
            print(
                f'\t... {name_attribution_method} attributions'
            )
            model = CNNModel(
                input_shape=input_shape,
                num_classes=num_labels,
                num_filters=model_config["num_filters"],
                filter_size=model_config["filter_size"],
                num_hidden_layers=model_config["num_hidden_layers"],
                dropout=model_config["dropout"]
            )

            if config['use_random_init']:
                print(
                    '/!\ Using random weight initializations for attribution'
                )

            else:

                if not torch.cuda.is_available():
                    model.load_state_dict(
                        torch.load(
                            model_path,
                            map_location=torch.device('cpu')
                        )
                    )

                else:
                    model.load_state_dict(torch.load(model_path))
            
            for subject in test_subjects:
                attributions_dir = os.path.join(
                    config["attributions_dir"],
                    name_attribution_method,
                    f'run-{run}',
                    f'sub_{subject}'
                )
                os.makedirs(
                    attributions_dir,
                    exist_ok=True
                )

                for image_path in test_image_paths[subject]:
                    image, label = load_data(
                        image_paths={subject: [image_path]},
                        return_fdata=True,
                        target_labeling=target_labeling[config["task"]]
                    )
                    image = np.expand_dims(
                        a=image,
                        axis=1
                    )
                    attribution = interpret_w_method(
                        model=model,
                        attribution_method=attribution_method,
                        image=image,
                        label=label,
                        num_labels=num_labels,
                        test_images=test_images,
                        device=device
                    )
                    attribution_img = nl.image.new_img_like(
                        ref_niimg=nb.load(image_path),
                        data=attribution[0,0],
                    )
                    attribution_img.to_filename(
                        filename=os.path.join(
                            attributions_dir,
                            f"{image_path.split('/')[-1].split('.')[0]}_attributions.nii.gz"
                        )
                    )
                        
            print(
                '\tDone!'
            )

        print(
            'Done!'
        )


def interpret_w_method(
    model,
    attribution_method,
    image,
    label,
    num_labels,
    test_images,
    device: torch.device=torch.device('cpu')
    ):
    name_attribution_method = str(attribution_method.__name__)
    
    if label.ndim > 1:
        label = label.ravel()[0]
    
    attribution = None
    label = int(label)
    image = torch.Tensor(image).to(torch.float)
    model.eval()
    
    if torch.cuda.is_available():
        model.to(device)

    if torch.cuda.is_available():
        image = image.to(device)

    image.requires_grad = True

    if name_attribution_method == 'LRP':
        grad_dummy = torch.eye(num_labels)[[label]]

        if torch.cuda.is_available():
            grad_dummy = grad_dummy.to(device)

        with attribution_method.context(model) as modified_model:
            output = modified_model(image)
            attribution, = torch.autograd.grad(
                output,
                image,
                grad_outputs=grad_dummy
            )
        
    elif name_attribution_method in {
        'DeepLift',
        'IntegratedGradients'
    }:
        attributer = attribution_method(model)
        baselines = (
            torch.zeros_like(image),
            torch.tensor(test_images.mean(axis=0, keepdims=True)).to(torch.float),
        )
        attribution = []

        for b in baselines:

            if torch.cuda.is_available():
                b = b.to(device)

            attribution.append(
                attributer.attribute(
                    inputs=image,
                    target=label,
                    baselines=(b)
                )
            )

        attribution = torch.stack(attribution)
        attribution = attribution.mean(dim=0)

    elif name_attribution_method == 'DeepLiftShap':
        attributer = attribution_method(model)
        baselines_idx = np.random.choice(
            test_images.shape[0],
            50,
            replace=False
        )
        attribution = attributer.attribute(
            inputs=image,
            target=label,
            baselines=torch.tensor(test_images[baselines_idx]).to(torch.float)
        )

    elif name_attribution_method == 'GuidedGradCam':
        last_conv_layer = None
        
        for module in model.modules():
            for layer in module.modules():
                if isinstance(layer, torch.nn.Conv3d):
                    last_conv_layer = layer
        
        attributer = attribution_method(model, last_conv_layer)
        attribution = attributer.attribute(
            inputs=image,
            target=label
        )

    elif name_attribution_method == 'SmoothGrad':
        attributer = attribution_method(model)
        attribution = attributer.attribute(
            inputs=image,
            nt_type='smoothgrad',
            nt_samples=50,
            target=label
        )

    else:
        attributer = attribution_method(model)
        attribution = attributer.attribute(
            inputs=image,
            target=label
        )
    
    if attribution is None:
        raise ValueError('attribution should be tensor but is None')

    return attribution.detach().cpu().numpy()


def attribute_argsparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='attribute model decoding decisions in given task'
    )
    parser.add_argument(
        '--task',
        metavar='DIR',
        default='WM',
        type=str,
        required=False,
        help='task for which data is interpreted (default: WM)'
    )
    parser.add_argument(
        '--fitted-model-dir',
        metavar='DIR',
        type=str,
        required=True,
        help='path where fitting runs for model that '
             'is to be interpreted are stored.'
             '(as resulting from running scripts/train.py)'
             'If multiple fitting runs exist for the model, '
             'the decoding decisions of the final model of each run'
             'are interpreted for the test data of the given task'
    )
    parser.add_argument(
        '--data-dir',
        metavar='DIR',
        type=str,
        default='data/task-WM',
        required=True,
        help='path where subject trial-images are stored'
             '(default: data/task-WM)'
    )
    parser.add_argument(
        '--attributions-dir',
        metavar='DIR',
        default='results/attributions/task-WM',
        type=str,
        required=False,
        help='path where attributions are stored (default: results/attributions)'
    )
    parser.add_argument(
        '--use-random-init',
        metavar='BOOL',
        default='False',
        type=str,
        choices=('True', 'False'),
        required=False,
        help='path where attributions are stored (default: results/attributions)'
    )
    parser.add_argument(
        '--seed',
        metavar='INT',
        default=12345,
        type=int,
        required=False,
        help='random seed (default: 1234)'
    )
    return parser

if __name__ == '__main__':
    
    interpret()