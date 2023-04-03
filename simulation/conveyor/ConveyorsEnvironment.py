import copy
import random
from logging import Logger
from typing import *

import torch
from simpy import Environment, Event, Interrupt

from agents import TorchAgent
import utils
from ml.utils import TensorWithMask
from simulation.conveyor.energy import consumption_Zhang
from simulation.conveyor.utils import WorldEvent, BagAppearanceEvent, UnsupportedEventType, Bag
from simulation.conveyor.model import ConveyorModel, all_unresolved_events, all_next_events
from topology import BaseTopology
from topology.utils import Section, conveyor_adj_nodes, conveyor_idx, node_type, node_id, node_conv_pos, \
    conveyor_adj_nodes_with_data, conv_start_node, conv_next_node, get_node_by_id


class ConveyorsEnvironment:
    """
    Environment which models the conveyor system and the movement of bags.
    """

    def __init__(self, config: Dict[str, Any], world_env: Environment, topology: BaseTopology, agent: TorchAgent,
                 logger: Logger):
        self._topology_config = config["topology"]
        self._world_env = world_env
        self._conveyors_move_proc = None
        self._current_bags = {}
        self._topology_graph = topology
        self._logger = logger

        conv_ids = [int(k) for k in self._topology_config["conveyors"].keys()]
        self._conveyor_models = {}
        for conv_id in conv_ids:
            checkpoints = conveyor_adj_nodes_with_data(self._topology_graph.graph, conv_id,
                                                       only_own=True, data="conveyor_pos")
            length = self._topology_config["conveyors"][str(conv_id)]["length"]
            model = ConveyorModel(self._world_env, length, checkpoints, model_id=conv_id, logger=logger)
            self._conveyor_models[conv_id] = model

        self._conveyor_upstreams = {
            conv_id: conveyor_adj_nodes(self._topology_graph.graph, conv_id)[-1]
            for conv_id in conv_ids
        }

        diverters_ids = [int(k) for k in self._topology_config["diverters"].keys()]
        self._diverter_agents = {}
        for dv_id in diverters_ids:
            self._diverter_agents[dv_id] = copy.deepcopy(agent)

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
            return self._checkInterrupt(lambda: self._putBagOnConveyor(conv_idx, bag, src))
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
        # self._logger.debug(
        #     f"Diverter {diverter} kicked bag {n_bag._id} from conveyor {conv_idx} to conveyor {up_conv}.")

    def _diverterPrediction(self, node: Section, bag: Bag, conv_idx: int):
        dv_id = node_id(node)
        dv_agent = self._diverter_agents[dv_id]

        dv_cfg = self._topology_config["diverters"][str(dv_id)]
        up_conv = dv_cfg["upstream_conv"]
        up_conv_node = conv_start_node(self._topology_graph.graph, up_conv)
        next_node = conv_next_node(self._topology_graph.graph, conv_idx, node)
        sink_node = next((s for s in self._topology_graph.sinks if s.id == bag._dst_id), None)
        assert sink_node is not None, "Sink node should be found"
        sample = self._topology_graph.get_sample(node, [up_conv_node, next_node], sink_node)

        sample_tensor = {}
        for key in sample.keys():
            if key == "neighbors_node_ids":
                sample_tensor[key] = TensorWithMask(
                    values=torch.tensor([sample[key]], dtype=torch.int64),
                    lengths=torch.tensor([len(sample[key])], dtype=torch.int64)
                )
            else:
                sample_tensor[key] = torch.tensor([sample[key]], dtype=torch.int64)

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

        pos = node_conv_pos(self._topology_graph.graph, conv_idx, node)
        assert pos is not None, "Position of the conveyor can't be None"

        model = self._conveyor_models[conv_idx]
        model.putObject(bag._id, bag, pos, return_nearest=False)

        bag.last_conveyor = conv_idx
        self._current_bags[bag._id] = set()
        self._current_bags[bag._id].add(node)

    def _leaveConveyorEnd(self, conv_idx: int, bag_id: int) -> bool:
        bag = self._removeBagFromConveyor(conv_idx, bag_id)
        up_node = self._conveyor_upstreams[conv_idx]
        up_type = node_type(up_node)

        if up_type == "sink":
            self._current_bags.pop(bag._id)
            self._logger.debug(f"Bag {bag._id} arrived to {up_node}.")
            current_time = self._world_env.now
            utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.add_scalar(
                f'Bag time/Bag arrived time',
                current_time - bag._start_time,
                current_time
            )
            return True

        if up_type == "junction":
            up_conv = conveyor_idx(self._topology_graph.graph, up_node)
            self._putBagOnConveyor(up_conv, bag, up_node)
        else:
            raise Exception("Invalid conveyor upstream node type: " + up_type)
        return False

    def _removeBagFromConveyor(self, conv_idx: int, bag_id: int):
        model = self._conveyor_models[conv_idx]
        bag = model.removeObject(bag_id)
        return bag

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

    def _updateEnergyConsumption(self):
        """
        Update energy consumption
        """
        cur_time = self._world_env.now
        time_diff = cur_time - self._energy_consumption_last_update
        self._energy_consumption_last_update = cur_time
        for _, model in self._conveyor_models.items():
            new_energy_consumption = consumption_Zhang(model._length, 1, len(model._objects)) * time_diff
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

            if bag._id in left_to_sinks or node in self._current_bags[bag._id]:
                continue

            atype = node_type(node)

            if atype == "conv_end":
                left_to_sink = self._leaveConveyorEnd(conv_idx, bag._id)
                if left_to_sink:
                    left_to_sinks.add(bag._id)
            elif atype == "diverter":
                forward_node = self._diverterPrediction(node, bag, conv_idx)

                if forward_node.type == "diverter":
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
