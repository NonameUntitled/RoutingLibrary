from agents import BaseAgent
from agents.ppo import PPOAgent
from utils import parse_args


def train():
    params = parse_args()

    # BaseAgent.create_from_config(params['agent_1'])  # random
    # BaseAgent.create_from_config(params['agent_2'])  # dqn
    # BaseAgent.create_from_config(params['agent_3'])  # reinforce
    PPOAgent.create_from_config(params['agent_4'])  # ppo


if __name__ == '__main__':
    train()
