import random

from simpy import Environment, Event, Interrupt
from collections import namedtuple, defaultdict

from simulation.messages import WorldEvent, BagAppearanceEvent, UnsupportedEventType, AgentId
from simulation.model import ConveyorModel, all_unresolved_events, all_next_events
from simulation.utils import conveyor_adj_nodes, node_conv_pos, conveyor_idx, agent_type, agent_idx, \
    make_conveyor_topology_graph
from topology.base import Section


class ConveyorsEnvironment:
    """
    Environment which models the conveyor system and the movement of bags.
    """

    def __init__(self, config, env: Environment, topology, agent):
        self.run_params = config['topology']
        self.env = env
        self.conveyors_move_proc = None
        self.current_bags = {}
        self.topology_graph = topology

        # dyn_env = DynamicEnv(time=lambda: self.env.now)
        conv_ids = [int(k) for k in self.run_params["conveyors"].keys()]
        self.conveyor_models = {}
        for conv_id in conv_ids:
            checkpoints = conveyor_adj_nodes(self.topology_graph, conv_id,
                                             only_own=True, data='conveyor_pos')
            length = self.run_params["conveyors"][str(conv_id)]['length']
            model = ConveyorModel(self.env, length, checkpoints, model_id=('world_conv', conv_id))
            self.conveyor_models[conv_id] = model

        self.conveyor_upstreams = {
            conv_id: conveyor_adj_nodes(self.topology_graph, conv_id)[-1]
            for conv_id in conv_ids
        }

        self._updateAll()

    def handleEvent(self, event: WorldEvent) -> Event:
        """
        Method which governs how events influence the environment.
        """
        if isinstance(event, BagAppearanceEvent):
            src = Section(type='source', id=event.src_id, position=0)
            bag = event.bag
            self.current_bags[bag.id] = set()
            conv_idx = conveyor_idx(self.topology_graph, src)
            return self._checkInterrupt(lambda: self._putBagOnConveyor(conv_idx, bag, src))
        else:
            raise UnsupportedEventType(event)

    def _diverterKick(self, dv_id: AgentId):
        """
        Checks if some bag is in front of a given diverter now,
        if so, moves this bag from current conveyor to upstream one.
        """
        assert agent_type(dv_id) == 'diverter', "Only diverter can kick"

        dv_idx = agent_idx(dv_id)
        dv_cfg = self.run_params['diverters'][str(dv_idx)]
        conv_idx = dv_cfg['conveyor']
        up_conv = dv_cfg['upstream_conv']
        pos = dv_cfg['pos']

        conv_model = self.conveyor_models[conv_idx]
        n_bag, n_pos = conv_model.nearestObject(pos)

        self._removeBagFromConveyor(conv_idx, n_bag.id)
        self._putBagOnConveyor(up_conv, n_bag, dv_id)
        print("Diverter %d kicked bag %d from conveyor %s to conveyor %s." % (
            dv_id[1], n_bag.id, str(conv_idx), str(up_conv)))

    def _putBagOnConveyor(self, conv_idx, bag, node):
        """
        Puts a bag on a given position to a given conveyor. If there is currently
        some other bag on a conveyor, throws a `CollisionException`
        """

        pos = node_conv_pos(self.topology_graph, conv_idx, node)
        assert pos is not None, "Position of the conveyor can't be None"

        model = self.conveyor_models[conv_idx]
        model.putObject(bag.id, bag, pos, return_nearest=False)

        bag.last_conveyor = conv_idx
        self.current_bags[bag.id] = set()
        self.current_bags[bag.id].add(node)

    def _leaveConveyorEnd(self, conv_idx, bag_id) -> bool:
        bag = self._removeBagFromConveyor(conv_idx, bag_id)
        up_node = self.conveyor_upstreams[conv_idx]
        up_type = agent_type(up_node)

        if up_type == 'sink':
            self.current_bags.pop(bag.id)
            print("Bag %d arrived to %s." % (bag.id, str(up_node)))
            return True

        if up_type == 'junction':
            up_conv = conveyor_idx(self.topology_graph, up_node)
            self._putBagOnConveyor(up_conv, bag, up_node)
        else:
            raise Exception('Invalid conveyor upstream node type: ' + up_type)
        return False

    def _removeBagFromConveyor(self, conv_idx, bag_id):
        model = self.conveyor_models[conv_idx]
        bag = model.removeObject(bag_id)
        return bag

    def _checkInterrupt(self, callback):
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

            for model in self.conveyor_models.values():
                model.pause()

            callback()
            self._updateAll()

        return Event(self.env).succeed()

    def _updateAll(self):
        """
        Check all current events and resolve them. After that continue moving
        """
        self.conveyors_move_proc = None

        left_to_sinks = set()
        # Resolving all immediate events
        for (conv_idx, (bag, node, delay)) in all_unresolved_events(self.conveyor_models):
            assert delay == 0, "Event delay should be zero"

            if bag.id in left_to_sinks or node in self.current_bags[bag.id]:
                continue

            atype = agent_type(node)

            if atype == 'conv_end':
                left_to_sink = self._leaveConveyorEnd(conv_idx, bag.id)
                if left_to_sink:
                    left_to_sinks.add(bag.id)
            elif atype == 'diverter':
                if (bool(random.getrandbits(1))):
                    self._diverterKick(node)
            elif atype != 'junction':
                raise Exception(f'Impossible conv node: {node}')

            if bag.id in self.current_bags and bag.id not in left_to_sinks:
                self.current_bags[bag.id].add(node)

        for conv_idx, model in self.conveyor_models.items():
            if model.resolving():
                model.endResolving()
            model.resume()

        self.conveyors_move_proc = self.env.process(self._move())

    def _move(self):
        """
        Check all future events, yield timeout to this event and _updateAll to handle this events
        """
        try:
            events = all_next_events(self.conveyor_models)

            if len(events) > 0:
                conv_idx, (bag, node, delay) = events[0]
                assert delay > 0, "Next event delay can't be 0"
                yield self.env.timeout(delay)
            else:
                yield Event(self.env)

            for model in self.conveyor_models.values():
                model.pause()

            self._updateAll()
        except Interrupt:
            pass
