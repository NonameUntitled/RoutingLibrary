from typing import Union

import networkx as nx
import numpy as np
import scipy.sparse as sp

from utils import MetaParent


class BaseEmbedding(metaclass=MetaParent):
    """
    Abstract class for graph node embeddings.
    """

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], weight: str):
        raise NotImplementedError()

    def transform(self, nodes):
        raise NotImplementedError()


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

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], weight='weight'):
        if isinstance(graph, nx.DiGraph):
            graph = nx.relabel_nodes(graph, agent_idx)  # TODO [Vladimir Baikalov]: Recognize how it works
            A = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes), weight=weight)
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
        u, s, vt = sp.linalg.svds(S, k=self.dim // 2, v0=np.ones(A.shape[0]))

        X1 = np.dot(u, np.diag(np.sqrt(s)))
        X2 = np.dot(vt.T, np.diag(np.sqrt(s)))
        self._W = np.concatenate((X1, X2), axis=1)

    def transform(self, idx):
        assert self._W is not None, 'Embedding matrix isn\'t fitted yet'
        return self._W[idx]


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

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], weight='weight'):
        if isinstance(graph, np.ndarray):
            graph = nx.from_numpy_array(graph, create_using=nx.DiGraph)
            weight = 'weight'

        graph = nx.relabel_nodes(graph.to_undirected(), agent_idx)  # TODO [Vladimir Baikalov]: Recognize how it works

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

    def transform(self, idx):
        assert self._W is not None, 'Embedding matrix isn\'t fitted yet'
        return self._W[idx]
