from bisect import bisect_right, bisect_left
from logging import Logger
from typing import *

from simpy import Environment

from simulation.utils import merge_sorted
from topology.utils import Section
import utils

POS_ROUND_DIGITS = 3
SOFT_COLLIDE_SHIFT = 0.2


class CollisionException(Exception):
    """
    Thrown when objects on conveyor collide
    """

    def __init__(self, args):
        super().__init__("[{}] Objects {} and {} collided in position {}"
                         .format(args[3], args[0], args[1], args[2]))
        self.obj1 = args[0]
        self.obj2 = args[1]
        self.pos = args[2]
        self.handler = args[3]


class AutomataException(Exception):
    pass


def search_pos(ls: List[Dict[str, Any]], pos: float,
               preference: str = "nearest") -> Optional[Tuple[Any, float]]:
    assert len(ls) > 0, "Empty list!"
    assert preference in ("nearest", "next", "prev"), "Invalid preference!"
    positions = [l["position"] for l in ls]
    if preference != "prev" and pos <= positions[0]:
        return ls[0], 0
    if preference != "next" and pos >= positions[-1]:
        return ls[-1], len(ls) - 1
    if preference == "prev" and pos <= positions[0]:
        return None
    if preference == "next" and pos >= positions[-1]:
        return None

    if preference == "nearest":
        id = bisect_left(positions, pos)
        return ls[id], id
    elif preference == "next":
        id = bisect_right(positions, pos)
        return ls[id], id
    elif preference == "prev":
        id = bisect_left(positions, pos)
        return ls[id], id


def shift(objs, d):
    return [{"id": object_description["id"], "position": round(object_description["position"] + d, POS_ROUND_DIGITS)}
            for
            object_description in objs]


# Automata for simplifying the modelling of objects movement
_model_automata = {
    "static": {"resume": "moving", "change": "dirty"},
    "moving": {"pause": "static"},
    "dirty": {"change": "dirty", "start_resolving": "resolving"},
    "resolving": {"change": "resolving", "end_resolving": "resolved"},
    "resolved": {"resume": "moving"}
}


class ConveyorModel:
    """
    Datatype which allows for modeling the conveyor with
    objects moving on top of it.

    A conveyor is modeled as a line with some checkpoints
    placed along it. Only one checkpoint can occupy a given position.

    Objects can be placed on a conveyor. All the objects on a conveyor
    move with the same speed - the current speed of a conveyor. Current speed
    of a conveyor can be changed.

    We want to have the following operations:
    - put an object to a conveyor (checking for collisions)
    - remove an object from a conveyor
    - change the conveyor speed # TODO[Aleksandr Pakulev]: Implement
    - update the positions of objects as if time T has passed with a current
      conveyor speed (T might be negative, moving objects backwards)
    - calculate time T until earliest event of object reaching a checkpoint
      (or the end of the conveyor), and also return those checkpoint and object
    """

    def __init__(self, world_env: Environment, length: float, checkpoints: List[Dict[str, Union[int, Section]]],
                 model_id: int,
                 logger: Logger, collision_distance: float = 0.1):
        assert length > 0, "Conveyor length <= 0!"

        checkpoints = sorted(checkpoints, key=lambda p: p["position"])
        if len(checkpoints) > 0:
            assert checkpoints[0]["position"] >= 0, \
                f"First checkpoint should not be negative. Got {checkpoints[0]['position']} for conveyor {model_id}."
            assert checkpoints[-1]["position"] < length, \
                f"Last checkpoint should not be more than the conveyor length. " \
                f"Got {checkpoints[-1]['position']} for conveyor {model_id}."
            for i in range(len(checkpoints) - 1):
                assert checkpoints[i]["position"] < checkpoints[i + 1]["position"], \
                    f"Checkpoints with equal positions. Got {checkpoints[i]['position']} and " \
                    f"{checkpoints[i + 1]['position']} for conveyor {model_id}"

        # constants
        self._world_env = world_env
        self._logger = logger
        self._model_id = model_id
        self._checkpoints = checkpoints
        self._checkpoint_positions = {cp["node"]: cp["position"] for cp in checkpoints}
        self._length = length
        self._collision_distance = collision_distance

        # variables
        # TODO[Aleksandr Pakulev]: Work with object in a more clear way
        self._speed = 1
        self._broken_type = None  # broken | dependence | collision {time}
        self._objects = {}
        self._object_positions = []

        self._state = "static"
        self._resume_time = 0
        self._resolved_events = set()

    def _stateTransfer(self, action: str) -> None:
        try:
            self._state = _model_automata[self._state][action]
        except KeyError:
            raise AutomataException(
                f"{self._model_id}: Invalid action `{action}` in state `{self._state}`;\n  "
                f"speed - {self._speed}m/s\n  cps - {self._checkpoints};\n  objs - {self._object_positions}")

    def setSpeed(self, speed: float) -> None:
        self._speed = speed

    def stop_conveyor(self, stop_info: Dict[str, Any]) -> None:
        self._logger.debug(f"STOP {self._model_id}: {stop_info}")
        self.setSpeed(0)
        self._broken_type = stop_info

    def start_conveyor(self) -> None:
        self._logger.debug(f"START {self._model_id}: {self._broken_type}")
        self.setSpeed(1)
        self._broken_type = None

    def nearestObject(self, pos: float, after=None, speed=None, not_exact=False,
                      preference="nearest") -> Optional[Tuple[Any, float]]:
        if len(self._object_positions) == 0:
            return None

        if after is not None:
            if speed is None:
                speed = self._speed
            objs = shift(self._object_positions, after * speed)
        else:
            objs = self._object_positions

        res = search_pos(objs, pos, preference=preference)
        if res is not None:
            object_description, idx = res
            (oid, o_pos) = object_description["id"], object_description["position"]
            if not_exact and o_pos == pos:
                if preference == "prev":
                    idx -= 1
                elif preference == "next":
                    idx += 1
                else:
                    raise Exception("please dont use nearest with not exact")
                if idx < 0 or idx > len(objs):
                    return None
                oid, o_pos = objs[idx]
            return self._objects[oid], o_pos
        return None

    def putObject(self, obj_id: int, obj: Any, pos: float):
        assert obj_id not in self._objects, f"Object {obj_id} already exists on {self._model_id} conveyor"
        pos = round(pos, POS_ROUND_DIGITS)

        if len(self._objects) > 0:
            object_description, n_idx = search_pos(self._object_positions, pos)
            (n_obj_id, n_pos) = (object_description["id"], object_description["position"])
            nextO = None
            prevO = None
            if n_idx == len(self._object_positions) - 1 and n_pos < pos:
                nextO = None
                prevO = object_description
            elif n_idx == 0 and n_pos > pos:
                nextO = object_description
                prevO = None
            else:
                nextO = object_description
                prevO = self._object_positions[n_idx - 1]

            if nextO is not None:
                if abs(nextO["position"] - pos) < self._collision_distance:
                    self._logger.debug(f"{self._model_id}: TRUE COLLISION: #{obj_id} and #{n_obj_id} on {pos}")
                    return True

            if prevO is not None:
                if abs(prevO["position"] - pos) < self._collision_distance:
                    self._logger.debug(f"{self._model_id}: TRUE COLLISION: #{obj_id} and #{n_obj_id} on {pos}")
                    return True

            if n_pos < pos:
                n_idx += 1
        else:
            n_idx = 0

        self._objects[obj_id] = obj
        self._object_positions.insert(n_idx, {"id": obj_id, "position": pos})
        self._stateTransfer("change")

        return False

    def check_collision(self, obj_id: int, pos: float):
        assert obj_id not in self._objects, f"Object {obj_id} already exists on {self._model_id} conveyor"
        pos = round(pos, POS_ROUND_DIGITS)

        if len(self._objects) > 0:
            object_description, n_idx = search_pos(self._object_positions, pos)
            (n_obj_id, n_pos) = (object_description["id"], object_description["position"])
            nextO = None
            prevO = None
            if n_idx == len(self._object_positions) - 1 and n_pos < pos:
                nextO = None
                prevO = object_description
            elif n_idx == 0 and n_pos > pos:
                nextO = object_description
                prevO = None
            else:
                nextO = object_description
                prevO = self._object_positions[n_idx - 1]

            if nextO is not None:
                if abs(nextO["position"] - pos) < self._collision_distance:
                    return {"is_collision": True, "time": self._collision_distance - abs(nextO["position"] - pos)}

            if prevO is not None:
                if abs(prevO["position"] - pos) < self._collision_distance:
                    return {"is_collision": True, "time": self._collision_distance + abs(prevO["position"] - pos)}

            if n_pos < pos:
                n_idx += 1
        else:
            n_idx = 0
        return {"is_collision": False, "idx": n_idx}

    def removeObject(self, obj_id: int):
        pos_idx = [object_description["id"] for object_description in self._object_positions].index(obj_id)

        self._object_positions.pop(pos_idx)
        obj = self._objects.pop(obj_id)
        self._stateTransfer("change")
        return obj

    def shift(self, d):
        self._object_positions = shift(self._object_positions, d)

    def skipTime(self, time: float, clean_ends=True):
        if time == 0:
            return 0

        self._stateTransfer("change")
        d = time * self._speed
        if len(self._objects) == 0:
            return d

        self.shift(d)

        # TODO[Aleksandr Pakulev]: Do we need it?
        if clean_ends:
            while len(self._object_positions) > 0 and self._object_positions[0]["position"] < 0:
                object_description = self._object_positions.pop(0)
                self._logger.debug("WHATEVER1")
                self._objects.pop(object_description["id"])
            while len(self._object_positions) > 0 and self._object_positions[-1]["position"] > self._length:
                object_description = self._object_positions.pop()
                self._logger.debug(f"WHATEVER2 {object_description}, {self._broken_type}, {self._model_id}")
                self._objects.pop(object_description["id"])

        return d

    # TODO[Aleksandr Pakulev]: Instead of using list of unclear tuples,
    #  it will be better to create a class for each type of events
    def nextEvents(self, skip_immediate=True) -> List[Tuple[Any, Any, float]]:

        if self._speed == 0 and not (self._broken_type is not None and self._broken_type["type"] == "collision"):
            return []

        obj = self._objects.keys()
        obj_positions = [object_description for object_description in self._object_positions if
                         object_description["id"] in obj]

        cps = self._checkpoint_positions.keys()
        c_points = [cp for cp in self._checkpoints if cp["node"] in cps]
        c_points.append({"node": Section("conv_end", self._model_id, self._length), "position": self._length})

        def _skip_cond(obj_id, cp_idx, pos):
            cp = c_points[cp_idx]
            if (obj_id, cp["node"]) in self._resolved_events:
                return True
            return cp["position"] <= pos if skip_immediate else cp["position"] < pos

        cp_idx = 0
        events = []
        for object_description in obj_positions:
            assert object_description["position"] >= 0 and object_description["position"] <= self._length, \
                "`nextEvents` on conveyor with undefined object positions!"

            while cp_idx < len(c_points) and _skip_cond(object_description["id"], cp_idx,
                                                        object_description["position"]):
                cp_idx += 1

            if cp_idx < len(c_points):
                cp = c_points[cp_idx]
                obj = self._objects[object_description["id"]]
                if self._broken_type is not None and self._broken_type["type"] == "collision":
                    if cp["position"] - object_description["position"] == 0:
                        events.append((obj, cp["node"], 0))
                else:
                    diff = (cp["position"] - object_description["position"]) / self._speed
                    events.append((obj, cp["node"], diff))

        events.sort(key=lambda p: p[2])

        if self._broken_type is not None and self._broken_type["type"] == "collision" and not (None, None) in self._resolved_events:
            diff = self._broken_type["time"] - self._world_env.now
            # TODO[Aleksandr Pakulev]: kostyl
            return [(None, None, 0 if diff <= 0 else diff)] + events
        return events

    def pickUnresolvedEvent(self) -> Union[Tuple[Any, Any, float], None]:
        assert self.resolving(), "picking event for resolution while not resolving"
        evs = self.nextEvents(skip_immediate=False)
        if len(evs) == 0:
            return None

        obj, cp, diff = evs[0]
        if diff > 0:
            return None

        if obj is None:
            self._resolved_events.add((None, None))
        else:
            self._resolved_events.add((obj._id, cp))
        return obj, cp, diff

    def resume(self) -> None:
        self._stateTransfer("resume")
        self._resume_time = self._world_env.now

    def pause(self) -> None:
        self._stateTransfer("pause")
        time_diff = self._world_env.now - self._resume_time
        assert time_diff >= 0, "Pause before resume"
        self.skipTime(time_diff)

    def startResolving(self):
        self._stateTransfer("start_resolving")

    def endResolving(self) -> None:
        self._resolved_events = set()
        self._stateTransfer("end_resolving")

    def static(self) -> bool:
        return self._state == "static"

    def dirty(self) -> bool:
        return self._state == "dirty"

    def resolving(self) -> bool:
        return self._state == "resolving"

    def resolved(self) -> bool:
        return self._state == "resolved"

    def moving(self) -> bool:
        return self._state == "moving"


def all_next_events(models: Dict[int, ConveyorModel]):
    res = []
    for conv_idx, model in models.items():
        evs = model.nextEvents()
        res = merge_sorted(res, [(conv_idx, ev) for ev in evs],
                           using=lambda p: p[1][2])
    return res


def all_unresolved_events(models: Dict[int, ConveyorModel]):
    while True:
        had_some = False
        for conv_idx, model in models.items():
            if model.dirty():
                model.startResolving()
            if model.resolving():
                ev = model.pickUnresolvedEvent()
                if ev is not None:
                    yield (conv_idx, ev)
                    had_some = True
        if not had_some:
            break
