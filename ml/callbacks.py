import utils
from utils import MetaParent, create_logger

logger = create_logger(name=__name__)


class BaseCallback(metaclass=MetaParent):

    def __init__(self, model, optimizer):
        self._model = model
        self._optimizer = optimizer

    def __call__(self, inputs, step_num):
        raise NotImplementedError


class CompositeCallback(BaseCallback, config_name='composite'):
    def __init__(
            self,
            model,
            optimizer,
            callbacks
    ):
        super().__init__(model, optimizer)
        self._callbacks = callbacks

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            model=kwargs['model'],
            optimizer=kwargs['optimizer'],
            callbacks=[
                BaseCallback.create_from_config(cfg, **kwargs)
                for cfg in config['callbacks']
            ]
        )

    def __call__(self, inputs, step_num):
        for callback in self._callbacks:
            callback(inputs, step_num)


class LossCallback(BaseCallback, config_name='loss'):

    def __init__(
            self,
            model,
            optimizer,
            on_step,
            regime_prefix,
            loss_prefix
    ):
        super().__init__(model, optimizer)
        self._on_step = on_step
        self._regime_prefix = regime_prefix
        self._loss_prefix = loss_prefix

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            model=kwargs['model'],
            optimizer=kwargs['optimizer'],
            on_step=config['on_step'],
            regime_prefix=config['regime_prefix'],
            loss_prefix=config['loss_prefix']
        )

    def __call__(self, inputs, step_num):
        if step_num % self._on_step == 0:
            if utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER:
                utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.add_scalar(
                    '{}/{}'.format(self._regime_prefix, self._loss_prefix),
                    inputs.get(self._loss_prefix, 0.0),
                    step_num
                )
                utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.flush()
