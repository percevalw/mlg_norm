import logging
from contextlib import contextmanager

import torch

logger = logging.getLogger("nlstruct")

subset_list = []


@contextmanager
def no_subset():
    global subset_list
    redos = [undo_subset() for undo_subset in list(subset_list)]
    yield
    subset_list = [redo() for redo in redos]


def index_shape(tensor, index, sliced_shape):
    indexer = tuple(slice(None) if d != sliced_shape else index for d in tensor.shape)
    return tensor[indexer]


def index_shape_put(tensor, index, sliced_shape, values):
    indexer = tuple(slice(None) if d != sliced_shape else index for d in tensor.shape)
    tensor[indexer] = values


class slice_parameters:
    def __init__(self, obj, names, indexer, optimizer, device=None):
        subset_module_params = {}
        if isinstance(names, list):
            names = {n: 0 for n in names}
        assert isinstance(names, dict)

        def do_subset():
            subset_module_params.clear()
            for module_param_name, dim in names.items():
                module_param = getattr(obj, module_param_name)
                if module_param is None:
                    continue

                sliced_shape = module_param.shape[dim]

                optimizer_saved_state = None
                optimizer_subset_state = None
                if isinstance(module_param, torch.nn.Parameter):
                    subset_module_param = torch.nn.Parameter(module_param[tuple(slice(None) for _ in range(dim)) + (indexer,)].to(device), requires_grad=module_param.requires_grad)
                    optimizer_saved_state = None
                    optimizer_subset_state = None
                    if optimizer is not None and optimizer.state[module_param]:
                        optimizer_saved_state = optimizer.state[module_param]
                        for group in optimizer.param_groups:
                            group['params'] = [subset_module_param if x is module_param else x for x in group['params']]
                        optimizer_subset_state = {}
                        for optim_param_name, optim_param in optimizer_saved_state.items():
                            param_device = device or optim_param.device
                            if hasattr(optim_param, 'shape') and sliced_shape in optim_param.shape:
                                optimizer_subset_state[optim_param_name] = index_shape(optim_param, indexer, sliced_shape).to(param_device)
                            elif hasattr(optim_param, 'to'):
                                optimizer_subset_state[optim_param_name] = optim_param.to(param_device)
                            else:
                                optimizer_subset_state[optim_param_name] = optim_param
                        optimizer.state[subset_module_param] = optimizer_subset_state
                        del optimizer.state[module_param]

                else:
                    subset_module_param = module_param[tuple(slice(None) for _ in range(dim)) + (indexer,)]

                subset_module_params[module_param_name] = (subset_module_param,
                                                           module_param.device,
                                                           module_param,  # .detach().cpu(),
                                                           optimizer_saved_state,
                                                           optimizer_subset_state,
                                                           dim)
                setattr(obj, module_param_name, subset_module_param)

            subset_list.append(undo_subset)
            return undo_subset

        def undo_subset():
            for module_param_name, (subset_module_param,
                                    device,
                                    module_param_detached,
                                    optimizer_saved_state,
                                    optimizer_subset_state,
                                    dim) in subset_module_params.items():
                sliced_shape = subset_module_param.data.shape[dim]
                original_sliced_shape = module_param_detached.data.shape[dim]
                index_shape_put(module_param_detached.data, indexer, original_sliced_shape, subset_module_param.detach().to(module_param_detached.device))
                restored_param = module_param_detached  # torch.nn.Parameter(module_param_detached.to(device), requires_grad=subset_module_param.requires_grad)

                # Update old embeddings with new ones

                if optimizer_saved_state is not None:
                    for group in optimizer.param_groups:
                        group['params'] = [restored_param if x is subset_module_param else x for x in group['params']]

                    for optim_param_name, optim_param in optimizer_subset_state.items():
                        if hasattr(optim_param, 'shape') and sliced_shape in optim_param.shape:
                            subset_param = optimizer_subset_state[optim_param_name]
                            optimizer_subset_state[optim_param_name] = optimizer_saved_state[optim_param_name]
                            index_shape_put(optimizer_subset_state[optim_param_name], indexer, original_sliced_shape, subset_param.to(optimizer_subset_state[optim_param_name].device))
                        optimizer.state[restored_param] = optimizer_subset_state
                    del optimizer.state[subset_module_param]
                setattr(obj, module_param_name, restored_param)

            subset_list.remove(undo_subset)

            return do_subset

        self.undo_subset = do_subset()

    def __call__(self, *args, **kwargs):
        return self.undo_subset()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.undo_subset()
