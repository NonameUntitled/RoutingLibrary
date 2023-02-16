import torch


def get_activation_function(name: str, **kwargs):
    if name == 'relu':
        return torch.nn.ReLU()
    elif name == 'gelu':
        return torch.nn.GELU()
    elif name == 'elu':
        return torch.nn.ELU(alpha=float(kwargs.get('alpha', 1.0)))
    elif name == 'leaky':
        return torch.nn.LeakyReLU(negative_slope=float(kwargs.get('negative_slope', 1e-2)))
    elif name == 'sigmoid':
        return torch.nn.Sigmoid()
    elif name == 'tanh':
        return torch.nn.Tanh()
    elif name == 'softmax':
        return torch.nn.Softmax()
    elif name == 'softplus':
        return torch.nn.Softplus(beta=int(kwargs.get('beta', 1.0)), threshold=int(kwargs.get('threshold', 20)))
    elif name == 'softmax_logit':
        return torch.nn.LogSoftmax()
    else:
        raise ValueError('Unknown activation function name `{}`'.format(name))


class TensorWithMask:
    def __init__(self, values: torch.Tensor, lengths: torch.Tensor):
        self._values = values
        self._lengths = lengths

    @property
    def flatten_values(self):
        return self._values

    @property
    def padded_values(self):
        # TODO[Vladimir Baikalov]: Use caching here
        batch_size = self._lengths.shape[0]
        max_sequence_length = self._lengths.max().item()

        # Shape: [batch_size, max_sequence_length, embedding_dim]
        padded_values = torch.zeros(
            batch_size, max_sequence_length, *self._values.shape[1:],
            dtype=self._values.dtype, device=self._values.device
        )

        padded_values[self.mask] = self._values

        return padded_values

    @property
    def lengths(self):
        return self._lengths

    @property
    def mask(self):
        # TODO[Vladimir Baikalov]: Use caching here
        batch_size = self._lengths.shape[0]
        max_sequence_length = self._lengths.max().item()

        # Shape: [batch_size, max_sequence_length]
        mask = torch.arange(
            end=max_sequence_length,
            device=self._lengths.device
        )[None].tile([batch_size, 1]) < self._lengths[:, None]

        return mask.bool()

    def to(self, device):
        self._values = self._values.to(device)
        self._lengths = self._lengths.to(device)
        return self

    def __iter__(self):
        return map(TensorWithMask, zip(self._values, self._lengths))
