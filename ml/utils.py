import torch


def get_activation_function(name, **kwargs):
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
