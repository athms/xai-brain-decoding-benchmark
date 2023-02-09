#!/usr/bin/env python3

import os, sys
import typing as tp
import argparse
from neuroquery import fetch_neuroquery_model, NeuroQueryModel
sys.path.append('./')
from src.plotting import plot_brain_img, clear_matplotlib_fig_cache


# query for each mental state in each datasets
# that we will pass to neuroquery:
QUERIES = {
    'WM': {
        'body': 'body images',
        'faces': 'face perception',
        'places': 'place perception',
        'tools': 'tool images'
    },
    'MOTOR': {
        'f': 'foot',
        'h': 'hand',
        't': 'tongue movement'
    },
    'heat-rejection': {
        'heat': 'pain perception',
        'rejection': 'social rejection'
    }
}


def meta_analysis(config: tp.Dict=None):
    """script's main function; computes meta analysis for mental
    states of each task with neuroquery."""

    if config is None:
        config = vars(get_argparse().parse_args())

    encoder = NeuroQueryModel.from_data_dir(fetch_neuroquery_model())

    for task in QUERIES:
        os.makedirs(os.path.join(config['meta_maps_dir'],f'task-{task}'), exist_ok=True)

        for state, query in QUERIES[task].items():
            if not os.path.isfile(
                os.path.join(
                    config['meta_maps_dir'],
                    f'task-{task}',
                    f"{state}_meta_map.png"
                )
            ):
                print(
                    f'encoding "{query}" into brain map...'
                )
                map = encoder(query)['brain_map']
                map.to_filename(
                    os.path.join(
                        config['meta_maps_dir'],
                        f'task-{task}',
                        f"{state}_meta_map.nii.gz"
                    )
                )
                plot_brain_img(
                    img=map,
                    path=os.path.join(
                        config['meta_maps_dir'],
                        f'task-{task}',
                        f"{state}_meta_map.png"
                    ),
                    workdir=os.path.join(
                        config['meta_maps_dir'],
                        f'task-{task}',
                        'tmp/'
                    ),
                    threshold=None,
                    dpi=300
                )
                clear_matplotlib_fig_cache()
                print('...done.')


def get_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='compute meta analysis with neuroquery for mental states'
    )
    parser.add_argument(
        '--meta-maps-dir',
        metavar='STR',
        default='results/meta_analysis/',
        type=str,
        required=False,
        help='path where meta analysis maps are stored '
             '(default: results/meta_analysis)'
    )
    return parser


if __name__ == '__main__':
    meta_analysis()