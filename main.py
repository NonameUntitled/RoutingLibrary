from v2.agents import BaseAgent
from v2.utils import parse_args


def train():
    args = parse_args()

    params = args.params

    BaseAgent.create_from_config(params['agent_1'])  # random
    BaseAgent.create_from_config(params['agent_2'])  # dqn
    BaseAgent.create_from_config(params['agent_3'])  # reinforce


if __name__ == '__main__':
    train()
