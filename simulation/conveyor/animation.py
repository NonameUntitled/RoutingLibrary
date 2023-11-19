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

        # Dict from edges to edges view
        edge_dict = {}
        for edge in self._edges:
            source = None
            target = None
            # Find corresponding nodes for the edge
            for node in self._nodes:
                if node["id"] == edge["source"]:
                    source = node
                elif node["id"] == edge["target"]:
                    target = node
                else:
                    continue
            if source is None or target is None:
                print("Error: edge", edge, "has no corresponding nodes")
                continue
            # Draw the edge
            line, = ax.plot([source["a_x"], target["a_x"]], [-source["a_y"], -target["a_y"]], 'k-')
            # Draw the arrow on the end of the edge
            arrow = FancyArrowPatch((source["a_x"], -source["a_y"]), (target["a_x"], -target["a_y"]), arrowstyle='->',
                                    mutation_scale=20)
            ax.add_patch(arrow)
            # Save the edge view
            edge_dict[(edge["source"], edge["target"])] = (line, arrow)

        # Dict from nodes to nodes view
        nodes_dict = {}
        for node in self._nodes:
            point, = ax.plot(node["a_x"], -node["a_y"], 'ko', markersize=10)
            nodes_dict[node["id"]] = point

        # Get the position of the object on the edge
        def get_object_position(object_info):
            source = None
            target = None
            # Find corresponding nodes for the edge on which the object is
            for node in self._nodes:
                if node["id"] == object_info.source:
                    source = node
                if node["id"] == object_info.target:
                    target = node
            # If the edge is not found, we will not draw the object
            # on the point (0, 0) with the message in the console
            if source is None or target is None:
                print("Error: object", object_info, "has no corresponding edge")
                return {"x": 0, "y": 0}
            x = source["a_x"] + (target["a_x"] - source["a_x"]) * object_info.position
            y = -(source["a_y"] + (target["a_y"] - source["a_y"]) * object_info.position)
            return {"x": x, "y": y}

        objects_dict = {}
        reset_once_nodes = []

        def update(frame):
            current_objects = {obj.id: obj for obj in self._steps[frame].positions}

            # Redraw objects which are already on the plot
            # If the object is not in the current step, remove it from the plot
            for o_id, point in list(objects_dict.items()):
                if o_id in current_objects:
                    object_position = get_object_position(current_objects[o_id])
                    point.set_data(object_position["x"], object_position["y"])
                else:
                    point.remove()
                    del objects_dict[o_id]

            # Draw new objects
            for o_id, obj in current_objects.items():
                if o_id not in objects_dict:
                    object_position = get_object_position(obj)
                    point, = ax.plot(object_position["x"], object_position["y"], 'co')
                    objects_dict[o_id] = point

            # Reset nodes which were colored for one step
            for node in reset_once_nodes:
                nodes_dict[node].set_color('k')
            reset_once_nodes.clear()

            for action in self._steps[frame].actions:
                # Change the color of the edge
                if action.type == "edge":
                    nodes = action.id
                    for i in range(len(nodes) - 1):
                        (line, arrow) = edge_dict[(nodes[i], nodes[i + 1])]
                        line.set_color(action.color)
                        arrow.set_color(action.color)
                # Change the color of the node
                # If the duration is "once", save the node to reset its color on the next step
                elif action.type == "node":
                    node = action.id
                    nodes_dict[node].set_color(action.color)
                    if action.duration == "once":
                        reset_once_nodes.append(node)

        # Save animation
        ani = FuncAnimation(fig, update, frames=len(self._steps), interval=interval, repeat=False)
        if filename is not None:
            ani.save(filename, writer='imagemagick')
        else:
            plt.show()
