import pandas as pd

from agents import BaseAgent
from ml import BaseEmbedding

import utils
from utils import create_logger, fix_random_seed, parse_args
from utils import CHECKPOINT_DIR, DEVICE

import os
import torch
import json
from topology import BaseTopology
from utils import parse_args

from simulation import BaseSimulation

logger = create_logger(name=__name__)
seed_val = 42


def main():
    params = parse_args()
    fix_random_seed(seed_val)

    logger.debug('Environment run config: \n{}'.format(json.dumps(params, indent=2)))

    # Utils initialization
    if 'experiment_name' in params:
        utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER = \
            utils.tensorboard_writers.TensorboardWriter(params['experiment_name'])

    # Environment-related parts initialization
    topology = BaseTopology.create_from_config(params['topology'])

    # Ml-related part initialization
    if 'shared_embedder' in params:
        embeddings = BaseEmbedding.create_from_config(config=params['shared_embedder'])
        if getattr(embeddings, 'fit', False):
            # If embedder has to invoke `fit` before usage
            # TODO[Vladimir Baikalov]: Think about how to fit not shared embeddings
            embeddings.fit(topology.graph)

    agent = BaseAgent.create_from_config(params['agent'])
    # TODO[Vladimir Baikalov]: check loading
    if 'pretrain_model_name' in params:
        load_path = os.path.join(CHECKPOINT_DIR, '{}.pth'.format(params['pretrain_model_name']))
        state_dict = torch.load(load_path, map_location='cpu')
        model_checkpoint = state_dict['model']
        agent.load_state_dict(model_checkpoint)
        logger.debug('Loaded checkpoint from {}'.format(load_path))
    agent = agent.to(DEVICE)

    # Environment initialization
    # environment = BaseSimulation.create_from_config(
    #     params['environment'],
    #     topology=topology,
    #     agent=agent
    # )
    #
    # environment.run(params['schedule'])
    simulation = BaseSimulation.create_from_config(params, topology=topology, agent=agent, logger=logger)
    event_series = simulation.run()
    event_series.save_all_series("./results/1")
    # event_series.draw_all_series_with_existing("./results/1", "./results")

    # BaseAgent.create_from_config(params['agent_1'])  # random
    # BaseAgent.create_from_config(params['agent_2'])  # dqn
    # BaseAgent.create_from_config(params['agent_3'])  # reinforce


if __name__ == '__main__':
    main()
