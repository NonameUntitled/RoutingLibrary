from base import TorchAgent


class ReinforceAgent(TorchAgent, config_name='reinforce'):

    def __init__(self, parts):
        super().__init__()
        self._parts = parts

    @classmethod
    def create_from_config(cls, config):
        return cls(
            parts=[
                BaseEncoder.create_from_config(part)
                for part in config['parts']
            ]
        )

    def forward(self, inputs):
        for part in self._parts:
            inputs = part(inputs)
        return inputs