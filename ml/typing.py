from abc import abstractmethod, ABC

from torch import Tensor


class TensorWithMask:
    def __init__(self, tensor: Tensor, mask: Tensor):
        self.tensor = tensor
        self.mask = mask


class TensorLike:
    @abstractmethod
    def to_tensor(self) -> Tensor:
        pass


class Value(TensorLike, ABC):
    pass


class Reward(TensorLike, ABC):
    pass


class Policy(TensorLike, ABC):
    pass


class State(ABC):
    pass


class ThreeNodesState(State):
    def __init__(
            self, current_node: Tensor,
            neighbor_nodes: TensorWithMask,
            destination_node: Tensor
    ):
        self.current_node = current_node
        self.neighbor_nodes = neighbor_nodes
        self.destination_node = destination_node
