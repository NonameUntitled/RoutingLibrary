from typing import Union

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import torch

from utils import shared, MetaParent


class BaseEmbedding(metaclass=MetaParent):
    """
    Abstract class for graph node embeddings.
    """

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], weight: str = 'length'):
        raise NotImplementedError

    def __call__(self, nodes_ids):
        raise NotImplementedError


class TorchEmbedding(BaseEmbedding, torch.nn.Module):

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], weight: str = 'length'):
        raise NotImplementedError

    def __call__(self, nodes_ids):
        raise NotImplementedError


@shared
class HOPEEmbedding(BaseEmbedding, config_name='hope'):

    def __init__(
            self,
            embedding_dim: int,
            proximity: str = 'katz',
            beta: float = 0.01
    ):
        super().__init__()
        if embedding_dim % 2 != 0:
            print(
                f'HOPE supports only even embedding dimensions. '
                f'Changing `embedding_dim` from {embedding_dim} to {embedding_dim - 1}'
            )
            embedding_dim -= 1

        if proximity not in {'katz', 'common-neighbors', 'adamic-adar'}:
            raise ValueError(f'Invalid proximity measure: {proximity}')

        self._embedding_dim = embedding_dim
        self._proximity = proximity
        self._beta = beta
        self._W = None

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], weight='length'):
        if isinstance(graph, nx.DiGraph):
            # TODO [Vladimir Baikalov]: Recognize how it works (line below)
            # graph = nx.relabel_nodes(graph, lambda x: x.id)
            A = nx.convert_matrix.to_numpy_array(graph, nodelist=sorted(graph.nodes), weight=weight)
            n = graph.number_of_nodes()
        else:
            A = np.mat(graph)
            n = A.shape[0]

        if self._proximity == 'katz':
            M_g = np.eye(n) - self._beta * A
            M_l = self._beta * A
        elif self._proximity == 'common-neighbors':
            M_g = np.eye(n)
            M_l = A * A
        elif self._proximity == 'adamic-adar':
            M_g = np.eye(n)
            D = np.mat(np.diag([1 / (np.sum(A[:, i]) + np.sum(A[i, :])) for i in range(n)]))
            M_l = A * D * A
        else:
            raise ValueError(f'Invalid proximity measure: {self._proximity}')

        S = np.dot(np.linalg.inv(M_g), M_l)
        u, s, vt = sp.linalg.svds(S, k=self._embedding_dim // 2, v0=np.ones(A.shape[0]))

        X1 = np.dot(u, np.diag(np.sqrt(s)))
        X2 = np.dot(vt.T, np.diag(np.sqrt(s)))
        self._W = np.concatenate((X1, X2), axis=1)

    def __call__(self, idx):
        assert self._W is not None, 'Embedding matrix isn\'t fitted yet'
        v = self._W[idx]
        if len(idx) == 1:
            v = np.array([v])
        return torch.Tensor(v)


@shared
class LaplacianEigenmap(BaseEmbedding, config_name='laplacian'):

    def __init__(
            self,
            embedding_dim: int,
            renormalize_weights: bool = True,
            weight_transform: str = 'heat',
            temp: float = 1.0
    ):
        super().__init__()

        if weight_transform not in {'inv', 'heat'}:
            raise ValueError(f'Invalid weight transform: {weight_transform}')

        self._embedding_dim = embedding_dim
        self._renormalize_weights = renormalize_weights
        self._weight_transform = weight_transform
        self._temp = temp
        self._W = None

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], weight='length'):
        if isinstance(graph, np.ndarray):
            graph = nx.from_numpy_array(graph, create_using=nx.DiGraph)
            weight = 'length'

        # TODO [Vladimir Baikalov]: Recognize how it works (line below)
        # graph = nx.relabel_nodes(
        #     graph.to_undirected(),
        #     lambda x: x.id
        # )

        if weight is not None:
            if self._renormalize_weights:
                sum_w = sum([ps[weight] for _, _, ps in graph.edges(data=True)])
                avg_w = sum_w / len(graph.edges())
                for u, v, ps in graph.edges(data=True):
                    graph[u][v][weight] /= avg_w

            if self._weight_transform == 'inv':
                for u, v, ps in graph.edges(data=True):
                    graph[u][v][weight] = 1 / ps[weight]
            elif self._weight_transform == 'heat':
                for u, v, ps in graph.edges(data=True):
                    w = ps[weight]
                    graph[u][v][weight] = np.exp(-w * w)
            else:
                raise ValueError(f'Invalid weight transform: {self._weight_transform}')

        A = nx.to_scipy_sparse_matrix(graph, nodelist=sorted(graph.nodes),
                                      weight=weight, format='csr', dtype=np.float32)

        n, m = A.shape
        diags = A.sum(axis=1)
        D = sp.spdiags(diags.flatten(), [0], m, n, format='csr')
        L = D - A

        print(n, m, len(graph.nodes))

        # (Changed by Igor):
        # Added v0 parameter, the "starting vector for iteration".
        # Otherwise, the operation behaves nondeterministically, and as a result
        # different nodes may learn different embeddings. I am not speaking about
        # minor floating point errors, the problem was worse.

        # values, vectors = sp.linalg.eigsh(L, k=self.dim + 1, M=D, which='SM')
        values, vectors = sp.linalg.eigsh(L, k=self._embedding_dim + 1, M=D, which='SM', v0=np.ones(A.shape[0]))

        # End (Changed by Igor)

        self._X = vectors[:, 1:]

        if weight is not None and self._renormalize_weights:
            self._X *= avg_w
        # print(self._X.flatten()[:3])

    def __call__(self, idx):
        assert self._X is not None, 'Embedding matrix isn\'t fitted yet'
        return self._X[idx]


@shared
class LearnableEmbedding(TorchEmbedding, config_name='learnable'):

    def __init__(
            self,
            embedding_dim: int,
            vocabulary_size: int
    ):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._vocabulary_size = vocabulary_size

        self._embeddings = torch.nn.Embedding(
            num_embeddings=self._vocabulary_size,
            embedding_dim=self._embedding_dim
        )

    @classmethod
    def create_from_config(cls, config):
        return cls(
            embedding_dim=config['embedding_dim'],
            vocabulary_size=config['vocabulary_size']
        )

    def __call__(self, nodes_ids):
        return self._embeddings(nodes_ids)

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], weight: str = 'length'):
        # We don't need to fit anything since it is learnable embeddings
        pass
