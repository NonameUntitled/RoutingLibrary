import torch.nn as nn
from v2.utils import MetaParent


class BaseAgent(metaclass=MetaParent):
    pass


class TorchAgent(BaseAgent, nn.Module):

    def __init__(self):
        super().__init__()
