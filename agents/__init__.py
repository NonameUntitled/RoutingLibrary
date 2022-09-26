import torch.nn as nn
from utils.registry import MetaParent


class BaseAgent(metaclass=MetaParent):
    pass


class TorchAgent(BaseAgent, nn.Module):

    def __init__(self):
        super().__init__()
