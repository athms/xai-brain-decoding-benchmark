#!/usr/bin/env python3

import os
from . import data
from . import model
from . import plotting

__all__ = [
    'data',
    'model',
    'plotting',
    'target_labeling',
    'reset_wandb_env'
]

target_labeling = {
    'heat-rejection': {
        'rejection': 0,
        'heat': 1
    },
    'WM': {
        'body': 0,
        'faces': 1,
        'places': 2,
        'tools': 3
    },
    'MOTOR': {
        'lh': 0,
        'lf': 1,
        'rh': 2,
        'rf': 3,
        't': 4
    }
}

def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
        "WANDB_RUN_GROUP"
    }
    for k in os.environ:
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]
    