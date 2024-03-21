import torch


def index_shape(tensor, index, sliced_shape):
    indexer = tuple(slice(None) if d != sliced_shape else index for d in tensor.shape)
    return tensor[indexer]


def index_shape_put(tensor, index, sliced_shape, values):
    indexer = tuple(slice(None) if d != sliced_shape else index for d in tensor.shape)
    tensor[indexer] = values


class SliceContext:
    instance = None

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.sliced_parameters = []

    @classmethod
    def get_current(cls):
        if SliceContext.instance is None:
            raise ValueError("No slicing context available")
        return SliceContext.instance

    def __enter__(self):
        SliceContext.instance = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        SliceContext.instance = None
        self.undo_slices()

    def slice_param_(self, module, names, indexer):
        if isinstance(names, list):
            names = {n: 0 for n in names}

        for name, dim in names.items():
            param = getattr(module, name)
            if param is None:
                raise AttributeError()

            device = param.device
            sliced_shape = param.shape[dim]

            optimizer_saved_state = optimizer_sliced_state = None
            if isinstance(param, torch.nn.Parameter):
                sliced_param = torch.nn.Parameter(
                    param[tuple(slice(None) for _ in range(dim)) + (indexer,)].to(
                        device
                    ),
                    requires_grad=param.requires_grad,
                )
                optimizer_saved_state = self.optimizer.state[param]
                assert optimizer_saved_state
                for group in self.optimizer.param_groups:
                    group["params"] = [
                        sliced_param if x is param else x for x in group["params"]
                    ]
                optimizer_sliced_state = {}
                for (
                    optim_param_name,
                    optim_param,
                ) in optimizer_saved_state.items():
                    param_device = device or optim_param.device
                    if (
                        hasattr(optim_param, "shape")
                        and sliced_shape in optim_param.shape
                    ):
                        optimizer_sliced_state[optim_param_name] = index_shape(
                            optim_param, indexer, sliced_shape
                        ).to(param_device)
                    elif hasattr(optim_param, "to"):
                        optimizer_sliced_state[optim_param_name] = optim_param.to(
                            param_device
                        )
                    else:
                        optimizer_sliced_state[optim_param_name] = optim_param
                self.optimizer.state[sliced_param] = optimizer_sliced_state
                del self.optimizer.state[param]
            else:
                sliced_param = param[
                    tuple(slice(None) for _ in range(dim)) + (indexer,)
                ]

            self.sliced_parameters.append(
                (
                    module,
                    name,
                    sliced_param,
                    param.device,
                    param,
                    optimizer_saved_state,
                    optimizer_sliced_state,
                    indexer,
                    dim,
                )
            )
            setattr(module, name, sliced_param)

    def undo_slices(self):
        for (
            module,
            name,
            sliced_param,
            device,
            param,
            optimizer_saved_state,
            optimizer_sliced_state,
            indexer,
            dim,
        ) in self.sliced_parameters:
            sliced_shape = sliced_param.data.shape[dim]
            original_sliced_shape = param.data.shape[dim]
            index_shape_put(
                param.data,
                indexer,
                original_sliced_shape,
                sliced_param.detach().to(param.device),
            )
            restored_param = param

            # Update old embeddings with new ones
            if optimizer_saved_state is not None:
                for group in self.optimizer.param_groups:
                    group["params"] = [
                        restored_param if x is sliced_param else x
                        for x in group["params"]
                    ]

                for optim_param_name, optim_param in optimizer_sliced_state.items():
                    if (
                        hasattr(optim_param, "shape")
                        and sliced_shape in optim_param.shape
                    ):
                        sliced_optim_param = optimizer_sliced_state[optim_param_name]
                        optimizer_sliced_state[
                            optim_param_name
                        ] = optimizer_saved_state[optim_param_name]
                        index_shape_put(
                            optimizer_sliced_state[optim_param_name],
                            indexer,
                            original_sliced_shape,
                            sliced_optim_param.to(
                                optimizer_sliced_state[optim_param_name].device
                            ),
                        )
                    self.optimizer.state[restored_param] = optimizer_sliced_state
                del self.optimizer.state[sliced_param]
            setattr(module, name, restored_param)

        self.sliced_parameters = []
