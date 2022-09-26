from simulation.conveyor import Conveyor
# from .agents import BaseAgent
from simulation.simulation import Simulation
from utils import parse_args


def train():
    args = parse_args()

    params = args  # .params

    # BaseAgent.create_from_config(params['agent_1'])  # random
    # BaseAgent.create_from_config(params['agent_2'])  # dqn
    # BaseAgent.create_from_config(params['agent_3'])  # reinforce
    #conv = Conveyor.create_from_config(params['conveyor1'])
    simulation = Simulation.create_from_config(params['simulation'])
    print(simulation._conveyors)


if __name__ == '__main__':
    train()
