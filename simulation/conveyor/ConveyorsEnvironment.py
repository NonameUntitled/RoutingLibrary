import copy
from logging import Logger
from typing import *

import torch
from simpy import Environment, Event, Interrupt

import utils
from agents import TorchAgent
from ml.utils import TensorWithMask
from simulation.conveyor.energy import consumption_Zhang, acceleration_consumption_Zhang, deceleration_consumption_Zhang
from simulation.conveyor.utils import WorldEvent, BagAppearanceEvent, UnsupportedEventType, Bag, ConveyorBreakEvent, \
    ConveyorRestoreEvent
from simulation.conveyor.model import ConveyorModel, all_unresolved_events, all_next_events
from simulation.conveyor.utils import WorldEvent, BagAppearanceEvent, UnsupportedEventType, Bag
from topology import BaseTopology
from topology.utils import Section, conveyor_adj_nodes, conveyor_idx, node_type, node_id, node_conv_pos, \
    conveyor_adj_nodes_with_data, conv_start_node, conv_next_node, get_node_by_id, only_reachable_from, working_topology
from utils.bag_trajectory import BaseBagTrajectoryMemory


class ConveyorsEnvironment:
    """
    Environment which models the conveyor system and the movement of bags.
    """

    def __init__(self, config: Dict[str, Any], world_env: Environment, topology: BaseTopology, agent: TorchAgent,
                 logger: Logger):
        self._topology_config = config["topology"]
        self._test_config = config["test"]
        self._learn_trigger_bag_count = config["learn_trigger_bag_count"]
        self._world_env = world_env
        self._conveyors_move_proc = None
        self._current_bags = {}
        self._topology_graph = topology
        self._logger = logger

        self._wrong_dst_reward = -100
        self._right_dst_reward = 100

        self._collisions = 0

        conv_ids = [int(k) for k in self._topology_config["conveyors"].keys()]
        self._conveyor_models = {}
        for conv_id in conv_ids:
            checkpoints = conveyor_adj_nodes_with_data(self._topology_graph.graph, conv_id,
                                                       only_own=True, data="conveyor_pos")
            length = self._topology_config["conveyors"][str(conv_id)]["length"]
            model = ConveyorModel(self._world_env, length, checkpoints, model_id=conv_id, logger=logger,
                                  collision_distance=config["test"]["collision_distance"])
            self._conveyor_models[conv_id] = model

        self._conveyor_broken = {conv_id: False for conv_id in conv_ids}
        self._conveyor_broken_dependencies = {conv_id: [] for conv_id in conv_ids}

        self._conveyor_upstreams = {
            conv_id: conveyor_adj_nodes(self._topology_graph.graph, conv_id)[-1]
            for conv_id in conv_ids
        }

        diverters_ids = [int(k) for k in self._topology_config["diverters"].keys()]
        self._diverter_agents = {}
        for dv_id in diverters_ids:
            agent_node_id = self._topology_graph._node_2_idx[
                Section(type="diverter", id=dv_id, position=int(self._topology_config["diverters"][str(dv_id)]["pos"]))
            ]
            self._diverter_agents[dv_id] = agent.copy(agent_node_id)

        self._lost_bags = 0
        self._arrived_bags = 0

        self._path_memory = BaseBagTrajectoryMemory.create_from_config(config['path_memory']) \
            if 'path_memory' in config else None

        # energy consumption
        self._system_energy_consumption = 0
        self._energy_consumption_last_update = self._world_env.now
        self._conveyor_energy_consumption = {}
        for conv_id in conv_ids:
            self._conveyor_energy_consumption[conv_id] = 0

        self._updateAll()

    def handleEvent(self, event: WorldEvent) -> Event:
        """
        Method which governs how events influence the environment.
        """
        if isinstance(event, BagAppearanceEvent):
            src = Section(type="source", id=event._src_id, position=0)
            bag = event._bag
            self._current_bags[bag._id] = set()
            conv_idx = conveyor_idx(self._topology_graph.graph, src)
            if self._conveyor_broken[conv_idx]:
                self._bag_lost_report(bag._id, f'Bag #{bag._id} came to the broken conveyor')
                return self._world_env.event()
            return self._checkInterrupt(lambda: self._putBagOnConveyor(conv_idx, bag, src))
        if isinstance(event, ConveyorBreakEvent):
            return self._checkInterrupt(lambda: self._conveyorBreak(event._conveyor_id))
        if isinstance(event, ConveyorRestoreEvent):
            return self._checkInterrupt(lambda: self._conveyorRestore(event._conveyor_id))
        else:
            raise UnsupportedEventType(event)

    def _diverterKick(self, diverter: Section, bag):
        """
        Checks if some bag is in front of a given diverter now,
        if so, moves this bag from current conveyor to upstream one.
        """
        assert node_type(diverter) == "diverter", "Only diverter can kick"

        dv_idx = node_id(diverter)
        dv_cfg = self._topology_config["diverters"][str(dv_idx)]
        conv_idx = dv_cfg["conveyor"]
        up_conv = dv_cfg["upstream_conv"]
        pos = dv_cfg["pos"]

        conv_model = self._conveyor_models[conv_idx]
        self._logger.debug(f"Bag {bag} an conveyor objects {conv_model._object_positions} and dicerter {diverter}.")
        n_bag, n_pos = conv_model.nearestObject(pos)

        self._removeBagFromConveyor(conv_idx, n_bag._id)
        self._putBagOnConveyor(up_conv, n_bag, diverter)
        self._logger.debug(
            f"Diverter {diverter} kicked bag {n_bag._id} from conveyor {conv_idx} to conveyor {up_conv}.")

    def _diverterPrediction(self, node: Section, bag: Bag, conv_idx: int):
        self._bag_time_reward_update(bag)
        dv_id = node_id(node)
        dv_agent = self._diverter_agents[dv_id]

        dv_cfg = self._topology_config["diverters"][str(dv_id)]
        up_conv = dv_cfg["upstream_conv"]
        up_conv_node = conv_start_node(self._topology_graph.graph, up_conv)
        next_node = conv_next_node(self._topology_graph.graph, conv_idx, node)
        sink_node = next((s for s in self._topology_graph.sinks if s.id == bag._dst_id), None)
        assert sink_node is not None, "Sink node should be found"

        current_topology_graph = working_topology(self._topology_graph.graph, self._conveyor_broken)
        topology_independence = self._test_config["topology_independence"]
        neighbor_nodes = [up_conv_node, next_node]
        if topology_independence == "full" or topology_independence == "only_nodes":
            neighbor_nodes = only_reachable_from(current_topology_graph, sink_node, neighbor_nodes)
        if topology_independence == "full" or topology_independence == "only_collisions":
            is_up_conv_available = self._conveyor_models[up_conv].check_collision(bag._id, 0)
            if is_up_conv_available["is_collision"]:
                neighbor_nodes = [next_node]
        if len(neighbor_nodes) == 0:
            neighbor_nodes = [next_node]
        sample = self._topology_graph.get_sample(node, neighbor_nodes, sink_node)

        # import networkx as nx
        #
        # nodes = self._topology_graph.graph.nodes
        #
        # rger = []
        #
        # for nn in nodes:
        #     rger.append(nx.has_path(self._topology_graph.graph, nn, sink_node))

        sample_tensor = {}
        for key in sample.keys():
            if key == "neighbors_node_ids":
                sample_tensor[key] = TensorWithMask(
                    values=torch.tensor(sample[key], dtype=torch.int64),
                    lengths=torch.tensor([len(sample[key])], dtype=torch.int64)
                )
            else:
                sample_tensor[key] = torch.tensor([sample[key]], dtype=torch.int64)

        sample_tensor[dv_agent._bag_ids_prefix] = torch.LongTensor([bag.id])

        output = dv_agent.forward(sample_tensor)
        forward_node_id = output[dv_agent._output_prefix].item()
        forward_node = get_node_by_id(self._topology_graph, forward_node_id)
        assert forward_node is not None, "Forward node should be found"
        return forward_node

    def _putBagOnConveyor(self, conv_idx: int, bag: Bag, node: Section):
        """
        Puts a bag on a given position to a given conveyor. If there is currently
        some other bag on a conveyor, throws a `CollisionException`
        """

        topology_independence = self._test_config["topology_independence"]
        # TODODO: smth here
        if self._conveyor_broken[conv_idx] and not (
                topology_independence == "full" or topology_independence == "only_collisions"):
            self._bag_lost_report(bag._id, f'Bag #{bag._id} came to the broken conveyor')
            return

        pos = node_conv_pos(self._topology_graph.graph, conv_idx, node)
        assert pos is not None, "Position of the conveyor can't be None"

        model = self._conveyor_models[conv_idx]
        is_collision = model.putObject(bag._id, bag, pos)

        # TODODO: smth here
        if is_collision:
            self._bag_collision_report()
            self._bag_lost_report(bag._id, f'Bag #{bag._id} lost because of collision')
            return

        bag.last_conveyor = conv_idx
        self._current_bags[bag._id] = set()
        self._current_bags[bag._id].add(node)

    def _conveyorBreak(self, conv_idx: int):
        """
        Breaks a conveyor.
        """
        model = self._conveyor_models[conv_idx]
        model.stop_conveyor({"type": "broken"})
        self._conveyorStartStopEnergyConsumption(conv_idx, "stop")

        self._conveyor_broken[conv_idx] = True
        self._logger.debug(f'Conveyor #{conv_idx} breaks')

    def _conveyorRestore(self, conv_idx: int):
        """
        Restores a conveyor.
        """
        model = self._conveyor_models[conv_idx]
        model.start_conveyor()

        for cnv in self._conveyor_broken_dependencies[conv_idx]:
            self._conveyorRestore(cnv)

        self._conveyor_broken_dependencies[conv_idx] = []

        self._conveyorStartStopEnergyConsumption(conv_idx, "stop")

        self._conveyor_broken[conv_idx] = False
        self._logger.debug(f'Conveyor #{conv_idx} restores')

    def _leaveConveyorEnd(self, conv_idx: int, bag_id: int) -> bool:
        up_node = self._conveyor_upstreams[conv_idx]
        up_type = node_type(up_node)

        if up_type == "sink":
            bag = self._removeBagFromConveyor(conv_idx, bag_id)
            if self._path_memory is not None:
                reward = self._wrong_dst_reward
                if bag._dst_id == up_node.id:
                    reward = self._right_dst_reward
                self._path_memory.add_reward_to_trajectory(bag._id, reward, 'time', terminal=True)
            self._bag_time_reward_update(bag)
            if up_node.id != bag._dst_id:
                self._bag_lost_report(bag._id,
                                      f'Bag #{bag._id} came to {up_node.id} sink, but its destination was {bag._dst_id}')
                return True
            else:
                self._logger.debug(f"Bag {bag._id} arrived to {up_node}.")
                current_time = self._world_env.now
                utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.add_scalar(
                    f'Bag time/Bag arrived time',
                    current_time - bag._start_time,
                    current_time
                )
                utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.add_scalar(
                    "Bag arrived/time",
                    self._arrived_bags + 1,
                    current_time
                )
                utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.flush()
                self._arrived_bags += 1
                self._current_bags.pop(bag._id)
                return True

        if up_type == "junction":
            up_conv = conveyor_idx(self._topology_graph.graph, up_node)
            pos = node_conv_pos(self._topology_graph.graph, up_conv, up_node)
            model = self._conveyor_models[conv_idx]
            up_model = self._conveyor_models[up_conv]
            # TODODO: check if it is correct
            topology_independence = self._test_config["topology_independence"]
            if topology_independence == "full" or topology_independence == "only_collisions":
                if self._conveyor_broken[up_conv]:
                    model.stop_conveyor({"type": "dependence"})
                    self._conveyor_broken[conv_idx] = True
                    self._conveyor_broken_dependencies[up_conv] = conv_idx
                    self._conveyorStartStopEnergyConsumption(conv_idx, "stop")
                    return False

                collision_check = up_model.check_collision(bag_id, pos)
                if collision_check["is_collision"]:
                    model.stop_conveyor({"type": "collision", "time": self._world_env.now + collision_check["time"]})
                    self._conveyor_broken[conv_idx] = True
                    self._conveyorStartStopEnergyConsumption(conv_idx, "stop")
                    return False

            bag = self._removeBagFromConveyor(conv_idx, bag_id)
            self._putBagOnConveyor(up_conv, bag, up_node)
        else:
            raise Exception("Invalid conveyor upstream node type: " + up_type)
        return False

    def _bag_time_reward_update(self, bag: Bag):
        current_time = self._world_env.now
        if self._path_memory is not None:
            self._path_memory.add_reward_to_trajectory(bag._id, bag._last_time_reward_time - current_time, 'time')
        bag.check_point(current_time)
        self._learn()

    def _learn(self):
        if not hasattr(self, '_learn_counter'):
            self._learn_counter = 0
        self._learn_counter += 1
        if self._learn_counter == self._learn_trigger_bag_count:
            if not hasattr(self, '_step_num'):
                self._step_num = 0
            for dv_agent in self._diverter_agents.values():
                loss = dv_agent.learn()
                if loss is not None:
                    if utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER:
                        utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.add_scalar(
                            '{}/{}'.format('train', f'agent_{dv_agent._node_id}'),
                            loss,
                            self._step_num
                        )
                        utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.flush()
            self._step_num += 1
            self._learn_counter = 0

    def _removeBagFromConveyor(self, conv_idx: int, bag_id: int):
        model = self._conveyor_models[conv_idx]
        return model.removeObject(bag_id)

    def _checkInterrupt(self, callback: Callable[[], None]) -> Event:
        """
        Pause conveyor to execute some action
        """
        if self.conveyors_move_proc is None:
            callback()
        else:
            try:
                self.conveyors_move_proc.interrupt()
                self.conveyors_move_proc = None
            except RuntimeError as err:
                pass

            for model in self._conveyor_models.values():
                model.pause()

            callback()
            self._updateAll()

        return Event(self._world_env).succeed()

    def _bag_collision_report(self):
        """
        Bag is collided
        """
        self._collisions += 1
        utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.add_scalar(
            f'Collisions/time',
            self._collisions,
            self._world_env.now
        )
        utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.flush()

    def _bag_lost_report(self, bag_id: int, message: str):
        """
        Bag is lost
        """
        utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.add_scalar(
            "Bag lost/time",
            self._lost_bags + 1,
            self._world_env.now
        )
        utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.flush()
        self._lost_bags += 1
        self._logger.debug(message)
        self._current_bags.pop(bag_id)

    def _conveyorStartStopEnergyConsumption(self, conv_idx: int, type: str):
        """
        Start conveyor energy consumption
        """

        model = self._conveyor_models[conv_idx]
        new_energy_consumption = acceleration_consumption_Zhang(model._length, model._speed,
                                                                len(model._objects)) if type == "start" else deceleration_consumption_Zhang(
            model._length, model._speed, len(model._objects))
        if len(model._objects) > 0:
            self._energy_reward_update(new_energy_consumption / len(model._objects),
                                       [obj._id for obj in model._objects.values()])

        self._system_energy_consumption += new_energy_consumption
        self._conveyor_energy_consumption[model._model_id] += new_energy_consumption
        utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.add_scalar(
            f'Conveyor {model._model_id} energy/time',
            self._conveyor_energy_consumption[model._model_id],
            self._energy_consumption_last_update
        )
        utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.flush()
        utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.add_scalar(
            "Conveyor system energy/time",
            self._system_energy_consumption,
            self._energy_consumption_last_update
        )
        utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.flush()

    def _updateEnergyConsumption(self):
        """
        Update energy consumption
        """
        cur_time = self._world_env.now
        time_diff = cur_time - self._energy_consumption_last_update
        self._energy_consumption_last_update = cur_time
        for _, model in self._conveyor_models.items():
            new_energy_consumption = consumption_Zhang(model._length, model._speed, len(model._objects)) * time_diff
            if len(model._objects) > 0:
                self._energy_reward_update(new_energy_consumption / len(model._objects),
                                           [obj._id for obj in model._objects.values()])

            self._system_energy_consumption += new_energy_consumption
            self._conveyor_energy_consumption[model._model_id] += new_energy_consumption
            utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.add_scalar(
                f'Conveyor {model._model_id} energy/time',
                self._conveyor_energy_consumption[model._model_id],
                self._energy_consumption_last_update
            )
            utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.flush()

        utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.add_scalar(
            "Conveyor system energy/time",
            self._system_energy_consumption,
            self._energy_consumption_last_update
        )
        utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.flush()

    def _energy_reward_update(self, energy_consumption_per_bag: float, bag_ids: List[int]):
        for bag_id in bag_ids:
            self._path_memory.add_reward_to_trajectory(bag_id, -energy_consumption_per_bag, 'energy')

    def _updateAll(self):
        """
        Check all current events and resolve them. After that continue moving
        """
        self._updateEnergyConsumption()

        self.conveyors_move_proc = None

        left_to_sinks = set()
        # Resolving all immediate events
        for (conv_idx, (bag, node, delay)) in all_unresolved_events(self._conveyor_models):
            assert delay == 0, "Event delay should be zero"

            if bag is None:
                self._conveyorRestore(conv_idx)
                continue

            if self._conveyor_broken[conv_idx]:
                continue

            if bag._id in left_to_sinks or node in self._current_bags[bag._id]:
                continue

            atype = node_type(node)

            if atype == "conv_end":
                left_to_sink = self._leaveConveyorEnd(conv_idx, bag._id)
                if left_to_sink:
                    left_to_sinks.add(bag._id)
                continue
            elif atype == "diverter":
                forward_node = self._diverterPrediction(node, bag, conv_idx)

                if forward_node.type == "diverter" and forward_node.position == 0:
                    self._diverterKick(node, bag)
            elif atype != "junction":
                raise Exception(f"Impossible conv node: {node}")

            if bag._id in self._current_bags and bag._id not in left_to_sinks:
                self._current_bags[bag._id].add(node)

        for conv_idx, model in self._conveyor_models.items():
            if model.resolving():
                model.endResolving()
            model.resume()

        self.conveyors_move_proc = self._world_env.process(self._move())

    def _move(self):
        """
        Check all future events, yield timeout to this event and _updateAll to handle this events
        """
        try:
            events = all_next_events(self._conveyor_models)

            if len(events) > 0:
                conv_idx, (bag, node, delay) = events[0]
                assert delay > 0, "Next event delay can't be 0"
                yield self._world_env.timeout(delay)
            else:
                yield Event(self._world_env)

            for model in self._conveyor_models.values():
                model.pause()

            self._updateAll()
        except Interrupt:
            pass
