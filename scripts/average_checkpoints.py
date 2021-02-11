#!/usr/bin/env python3
# coding: utf-8

"""
Checkpoint averaging 

Mainly follows:
https://github.com/pytorch/fairseq/blob/master/scripts/average_checkpoints.py
"""

import argparse
import collections
import torch
from typing import List


def average_checkpoints(inputs: List[str]) -> dict:
    """Loads checkpoints from inputs and returns a model with averaged weights.
    Args:
      inputs: An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    for f in inputs:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(
                    s, 'cpu')
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        # Averaging: only handle the network params. 
        model_params = state['model_state']

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    # v should be a list of torch Tensor.
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state['model_state'] = averaged_params
    return new_state


def main():
    parser = argparse.ArgumentParser(
        description='Tool to average the params of input checkpoints to '
                    'produce a new checkpoint',
    )
    parser.add_argument('--inputs', required=True, nargs='+',
                        help='Input checkpoint file paths.')
    parser.add_argument('--output', required=True, metavar='FILE',
                        help='Write the new checkpoint to this path.')
    args = parser.parse_args()
    print(args)

    new_state = average_checkpoints(args.inputs)
    torch.save(new_state, args.output)
    print('Finished writing averaged checkpoint to {}.'.format(args.output))


if __name__ == '__main__':
    main()

