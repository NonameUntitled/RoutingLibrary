from agents import BaseAgent
from ml import BaseCallback
from ml import BaseEmbedding
from ml import BaseLoss
from ml import BaseOptimizer
from ml.utils import collate_fn, TensorWithMask

from topology import BaseTopology

import utils
from utils import create_logger, fix_random_seed, parse_args
from utils import CHECKPOINT_DIR, DEVICE

import os
import torch
import json

from functools import partial

from torch.utils.data import DataLoader

logger = create_logger(name=__name__)
seed_val = 42


def pretrain(dataloader, model, loss_function, optimizer, callback, num_epochs):
    step_num = 0

    logger.debug('Start pre-training...')

    for epoch in range(num_epochs):
        logger.debug(f'Start pre-train epoch {epoch}')
        for step, batch in enumerate(dataloader):
            model.train()

            for key, values in batch.items():
                batch[key] = batch[key].to(DEVICE)

            model_output = model(batch)
            batch.update(model_output)

            for key, values in batch.items():
                if isinstance(values, TensorWithMask):
                    batch[key] = values.flatten_values

            loss = loss_function(batch)

            optimizer.step(loss)
            callback(batch, step_num)

            step_num += 1

    logger.debug('Pre-training procedure has been finished!')
    return model.state_dict()


def main():
    params = parse_args()
    fix_random_seed(seed_val)

    logger.debug('Pre-train config: \n{}'.format(json.dumps(params, indent=2)))

    # Utils initialization
    if 'experiment_name' in params:
        utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER = \
            utils.tensorboard_writers.TensorboardWriter(params['experiment_name'])

    # Environment-related part initialization
    topology = BaseTopology.create_from_config(params['topology'])
    # But it is possible only in yaml not in json
    # Another idea is to create separated field outside
    # TODO[Vladimir Baikalov]: It looks bad it is better to have sort of reference
    embeddings = BaseEmbedding.create_from_config(config=params['shared_embedder'])
    embeddings.fit(topology.graph)  # For some embedders this one is required, for some it can be skipped

    # Ml-related part initialization
    topology_data_schema = topology.schema
    train_dataset = topology.gen_episodes(params['num_samples'])
    validation_dataset = topology.gen_episodes(params['num_samples'])

    # TODO[Vladimir Baikalov]: Put dataloader params into config
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=1024,
        shuffle=True,
        drop_last=True,
        collate_fn=partial(collate_fn, schema=topology_data_schema)
    )
    # TODO[Vladimir Baikalov]: Use validation dataloader for metric monitoring
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=1024,
        drop_last=False,
        collate_fn=partial(collate_fn, schema=topology_data_schema)
    )

    agent = BaseAgent.create_from_config(params['agent_3']).to(DEVICE)
    loss = BaseLoss.create_from_config(params['loss_2'])
    optimizer = BaseOptimizer.create_from_config(params['optimizer'], model=agent)
    callback = BaseCallback.create_from_config(
        params['callback'],
        model=agent,
        optimizer=optimizer,
        validation_dataloader=validation_dataloader
    )

    # TODO[Vladimir Baikalov]: make sanity check of it on bigger topology
    # TODO[Vladimir Baikalov]: handle SIGKILL/SIGTERM/Ctrl + C
    final_model_state_dict = pretrain(
        dataloader=train_dataloader,
        model=agent,
        loss_function=loss,
        optimizer=optimizer,
        callback=callback,
        num_epochs=params['num_epochs']
    )

    # TODO[Vladimir Baikalov]: Instead of `model_name` implement to derive name from pipeline
    save_path = os.path.join(CHECKPOINT_DIR, '{}.pth'.format(params['model_name']))
    torch.save(
        {'model': final_model_state_dict, 'optimizer': optimizer.state_dict()},
        save_path
    )
    logger.debug('Saved model checkpoint as : {}'.format(save_path))


if __name__ == '__main__':
    main()
