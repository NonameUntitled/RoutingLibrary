import utils
from utils import MetaParent, create_logger

logger = create_logger(name=__name__)


class BaseCallback(metaclass=MetaParent):

    def __init__(self, model, dataloader, optimizer):
        self._model = model
        self._dataloader = dataloader
        self._optimizer = optimizer

    def __call__(self, inputs, step_num):
        raise NotImplementedError


class LossCallback(BaseCallback, config_name='loss'):

    def __init__(
            self,
            model,
            dataloader,
            optimizer,
            on_step,
            regime_prefix,
            loss_prefix
    ):
        super().__init__(model, dataloader, optimizer)
        self._on_step = on_step
        self._regime_prefix = regime_prefix
        self._loss_prefix = loss_prefix

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            model=kwargs['model'],
            dataloader=kwargs['dataloader'],
            optimizer=kwargs['optimizer'],
            on_step=config['on_step'],
            regime_prefix=config['regime_prefix'],
            loss_prefix=config['loss_prefix']
        )

    def __call__(self, inputs, step_num):
        if step_num % self._on_step == 0:
            utils.tensorboards.GLOBAL_TENSORBOARD_WRITER.add_scalar(
                '{}/{}'.format(self._regime_prefix, self._loss_prefix),
                inputs[self._loss_prefix],
                step_num
            )
            utils.tensorboards.GLOBAL_TENSORBOARD_WRITER.flush()
