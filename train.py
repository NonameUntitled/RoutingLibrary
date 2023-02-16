from agents import BaseAgent
from utils import parse_args

from simulation import BaseSimulation


def train():
    simulation_params, agent_params = parse_args()

    simulation = BaseSimulation.create_from_config(simulation_params['simulation'])
    agent = BaseAgent.create_from_config(agent_params['agent'])  # ppo

    simulation.run(agent)

    # BaseAgent.create_from_config(params['agent_1'])  # random
    # BaseAgent.create_from_config(params['agent_2'])  # dqn
    # BaseAgent.create_from_config(params['agent_3'])  # reinforce


if __name__ == '__main__':
    train()
