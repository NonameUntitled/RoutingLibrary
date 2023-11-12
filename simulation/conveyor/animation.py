from dataclasses import dataclass

import matplotlib

from topology.utils import Section

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

from topology import BaseTopology


@dataclass
class Action:
    # "edge" or "node",
    type: str
    # list of nodes for "edge" and node for "node"
    id: Section | list[Section]
    # new color of the object
    color: str
    # "once" or "continuous"
    duration: str


@dataclass
class ObjectPosition:
    # node ids
    source: Section
    target: Section
    # object id
    id: any
    # float from 0 to 1 which means the position of the object on the edge
    position: float


@dataclass
class Step:
    positions: list[ObjectPosition]
    actions: list[Action]


class Animation:
    def __init__(self, topology: BaseTopology):
        self._nodes = [{"id": s, "a_x": d["a_x"], "a_y": d["a_y"]} for (s, d) in topology.graph.nodes(data=True)]
        self._edges = [{"source": s, "target": t} for (s, t, d) in topology.graph.edges(data=True)]
        self._steps = []
        self._actions = []
        self._object_positions = []

    def commit_step(self):
        self._steps.append(Step(positions=self._object_positions, actions=self._actions))
        self._actions = []
        self._object_positions = []

    def add_action(self, action: Action):
        self._actions.append(action)

    def add_object_position(self, object_position: ObjectPosition):
        self._object_positions.append(object_position)

    def draw_animation(self, filename=None, interval=100):
        fig, ax = plt.subplots()

        edge_dict = {}
        for edge in self._edges:
            source = None
            target = None
            for node in self._nodes:
                if node["id"] == edge["source"]:
                    source = node
                elif node["id"] == edge["target"]:
                    target = node
                else:
                    continue
            if source is None or target is None:
                continue
            line, = ax.plot([source["a_x"], target["a_x"]], [-source["a_y"], -target["a_y"]], 'k-')
            arrow = FancyArrowPatch((source["a_x"], -source["a_y"]), (target["a_x"], -target["a_y"]), arrowstyle='->',
                                    mutation_scale=20)
            ax.add_patch(arrow)
            edge_dict[(edge["source"], edge["target"])] = (line, arrow)

        nodes_dict = {}
        for node in self._nodes:
            point, = ax.plot(node["a_x"], -node["a_y"], 'ko', markersize=10)
            nodes_dict[node["id"]] = point

        def get_bag_positions(bag_info):
            source = None
            target = None
            for node in self._nodes:
                if node["id"] == bag_info.source:
                    source = node
                if node["id"] == bag_info.target:
                    target = node
            if source is None or target is None:
                return {"x": 0, "y": 0}
            x = source["a_x"] + (target["a_x"] - source["a_x"]) * bag_info.position
            y = -(source["a_y"] + (target["a_y"] - source["a_y"]) * bag_info.position)
            return {"x": x, "y": y}

        objects_dict = {}
        reset_once_nodes = []

        def update(frame):
            current_objects = {obj.id: obj for obj in self._steps[frame].positions}

            for o_id, point in list(objects_dict.items()):
                if o_id in current_objects:
                    object_position = get_bag_positions(current_objects[o_id])
                    point.set_data(object_position["x"], object_position["y"])
                else:
                    point.remove()
                    del objects_dict[o_id]

            for o_id, obj in current_objects.items():
                if o_id not in objects_dict:
                    object_position = get_bag_positions(obj)
                    point, = ax.plot(object_position["x"], object_position["y"], 'co')
                    objects_dict[o_id] = point

            for node in reset_once_nodes:
                nodes_dict[node].set_color('k')
            reset_once_nodes.clear()

            for action in self._steps[frame].actions:
                if action.type == "edge":
                    nodes = action.id
                    for i in range(len(nodes) - 1):
                        (line, arrow) = edge_dict[(nodes[i], nodes[i + 1])]
                        line.set_color(action.color)
                        arrow.set_color(action.color)
                elif action.type == "node":
                    node = action.id
                    nodes_dict[node].set_color(action.color)
                    if action.duration == "once":
                        reset_once_nodes.append(node)

        # save animation
        ani = FuncAnimation(fig, update, frames=len(self._steps), interval=interval, repeat=False)
        if filename is not None:
            ani.save(filename, writer='imagemagick')
        else:
            plt.show()
