from typing import *

from simpy import Environment

from simulation.utils import binary_search, differs_from, merge_sorted
from topology.utils import Section

POS_ROUND_DIGITS = 3
SOFT_COLLIDE_SHIFT = 0.2


class CollisionException(Exception):
    """
    Thrown when objects on conveyor collide
    """

    def __init__(self, args):
        super().__init__('[{}] Objects {} and {} collided in position {}'
                         .format(args[3], args[0], args[1], args[2]))
        self.obj1 = args[0]
        self.obj2 = args[1]
        self.pos = args[2]
        self.handler = args[3]


class AutomataException(Exception):
    pass


def search_pos(ls: List[Tuple[Any, float]], pos: float,
               return_index: bool = False,
               preference: str = 'nearest') -> Tuple[Any, float]:
    assert len(ls) > 0, "what are you gonna find pal?!"
    return binary_search(ls, differs_from(pos, using=lambda p: p[1]),
                         return_index=return_index, preference=preference)


def shift(objs, d):
    return [(o, round(p + d, POS_ROUND_DIGITS)) for (o, p) in objs]


# Automata for simplifying the modelling of objects movement
_model_automata = {
    'pristine': {'resume': 'moving', 'change': 'dirty'},
    'moving': {'pause': 'pristine'},
    'dirty': {'change': 'dirty', 'start_resolving': 'resolving'},
    'resolving': {'change': 'resolving', 'end_resolving': 'resolved'},
    'resolved': {'resume': 'moving'}
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
    - change the conveyor speed
    - update the positions of objects as if time T has passed with a current
      conveyor speed (T might be negative, moving objects backwards)
    - calculate time T until earliest event of object reaching a checkpoint
      (or the end of the conveyor), and also return those checkpoint and object
    """

    def __init__(self, env: Environment, length: float, checkpoints: list[dict[str, int | Section]], model_id: int):
        assert length > 0, "Conveyor length <= 0!"

        checkpoints = sorted(checkpoints, key=lambda p: p['position'])
        if len(checkpoints) > 0:
            assert checkpoints[0]['position'] >= 0, "Checkpoints with position < 0!"
            assert checkpoints[-1]['position'] < length, "Checkpoint with position >= conveyor length!"
            for i in range(len(checkpoints) - 1):
                assert checkpoints[i]['position'] < checkpoints[i + 1]['position'], \
                    "Checkpoints with equal positions!"

        # constants
        self._env = env
        self._model_id = model_id
        self._checkpoints = checkpoints
        self._checkpoint_positions = {cp['node']: cp['position'] for cp in checkpoints}
        self._length = length

        # variables
        self._speed = 1
        self._objects = {}
        self._object_positions = []

        self._state = 'pristine'
        self._resume_time = 0
        self._resolved_events = set()

    def _stateTransfer(self, action: str) -> None:
        try:
            self._state = _model_automata[self._state][action]
        except KeyError:
            raise AutomataException(
                f'{self._model_id}: Invalid action `{action}` in state `{self._state}`;\n  speed - {self._speed}m/s\n  cps - {self._checkpoints};\n  objs - {self._object_positions}')


    def nearestObject(self, pos: float, after=None, speed=None, not_exact=False,
                      preference='nearest') -> Optional[Tuple[Any, float]]:
        if len(self._object_positions) == 0:
            return None

        if after is not None:
            if speed is None:
                speed = self._speed
            objs = shift(self._object_positions, after * speed)
        else:
            objs = self._object_positions

        res = search_pos(objs, pos, preference=preference, return_index=True)
        if res is not None:
            (oid, o_pos), idx = res
            if not_exact and o_pos == pos:
                if preference == 'prev':
                    idx -= 1
                elif preference == 'next':
                    idx += 1
                else:
                    raise Exception('please dont use nearest with not exact')
                if idx < 0 or idx > len(objs):
                    return None
                oid, o_pos = objs[idx]
            return self._objects[oid], o_pos
        return None

    def putObject(self, obj_id: int, obj: Any, pos: float, soft_collide=True, return_nearest=False):
        assert obj_id not in self._objects, "Clashing object ID!"
        pos = round(pos, POS_ROUND_DIGITS)

        nearest = None
        if len(self._objects) > 0:
            (n_obj_id, n_pos), n_idx = search_pos(self._object_positions, pos, return_index=True)
            if n_pos == pos:
                if soft_collide:
                    print('{}: TRUE COLLISION: #{} and #{} on {}'.format(self._model_id, obj_id, n_obj_id, pos))
                    i = n_idx
                    p_pos = pos
                    while i < len(self._object_positions) and self._object_positions[i][1] >= p_pos:
                        p_pos = round(p_pos + SOFT_COLLIDE_SHIFT, POS_ROUND_DIGITS)
                        self._object_positions[i] = (self._object_positions[i][0], p_pos)
                        i += 1
                else:
                    raise CollisionException((obj, self._objects[n_obj_id], pos, self._model_id))
            elif n_pos < pos:
                n_idx += 1
            nearest = (n_obj_id, n_pos)
        else:
            n_idx = 0

        self._objects[obj_id] = obj
        self._object_positions.insert(n_idx, (obj_id, pos))
        self._stateTransfer('change')

        if return_nearest:
            return nearest

    def removeObject(self, obj_id: int):
        pos_idx = None
        for (i, (oid, pos)) in enumerate(self._object_positions):
            if oid == obj_id:
                pos_idx = i
                break

        self._object_positions.pop(pos_idx)
        obj = self._objects.pop(obj_id)
        self._stateTransfer('change')
        return obj

    def shift(self, d):
        self._object_positions = shift(self._object_positions, d)

    def skipTime(self, time: float, clean_ends=True):
        if time == 0:
            return 0

        self._stateTransfer('change')
        d = time * self._speed
        if len(self._objects) == 0:
            return d

        self.shift(d)

        if clean_ends:
            while len(self._object_positions) > 0 and self._object_positions[0][1] < 0:
                obj_id, _ = self._object_positions.pop(0)
                self._objects.pop(obj_id)
            while len(self._object_positions) > 0 and self._object_positions[-1][1] > self._length:
                obj_id, pos = self._object_positions.pop()
                self._objects.pop(obj_id)

        return d

    def nextEvents(self, skip_immediate=True) -> List[Tuple[Any, Any, float]]:
        if self._speed == 0:
            return []

        obj = self._objects.keys()
        obj_positions = [(oid, pos) for (oid, pos) in self._object_positions
                         if oid in obj]

        cps = self._checkpoint_positions.keys()
        c_points = [cp for cp in self._checkpoints if cp['node'] in cps]
        c_points.append({'node': Section('conv_end', self._model_id, self._length), 'position': self._length})

        def _skip_cond(obj_id, cp_idx, pos):
            cp = c_points[cp_idx]
            if (obj_id, cp['node']) in self._resolved_events:
                return True
            return cp['position'] <= pos if skip_immediate else cp['position'] < pos

        cp_idx = 0
        events = []
        for (obj_id, pos) in obj_positions:
            assert pos >= 0 and pos <= self._length, \
                "`nextEvents` on conveyor with undefined object positions!"

            while cp_idx < len(c_points) and _skip_cond(obj_id, cp_idx, pos):
                cp_idx += 1

            if cp_idx < len(c_points):
                cp = c_points[cp_idx]
                obj = self._objects[obj_id]
                diff = (cp['position'] - pos) / self._speed
                events.append((obj, cp['node'], diff))

        events.sort(key=lambda p: p[2])
        return events

    def pickUnresolvedEvent(self) -> Union[Tuple[Any, Any, float], None]:
        assert self.resolving(), "picking event for resolution while not resolving"
        evs = self.nextEvents(skip_immediate=False)
        if len(evs) == 0:
            return None

        obj, cp, diff = evs[0]
        if diff > 0:
            return None

        self._resolved_events.add((obj._id, cp))
        return obj, cp, diff

    def resume(self) -> None:
        self._stateTransfer('resume')
        self._resume_time = self._env.now

    def pause(self) -> None:
        self._stateTransfer('pause')
        time_diff = self._env.now - self._resume_time
        assert time_diff >= 0, "Pause before resume"
        self.skipTime(time_diff)

    def startResolving(self):
        self._stateTransfer('start_resolving')

    def endResolving(self) -> None:
        self._resolved_events = set()
        self._stateTransfer('end_resolving')

    def pristine(self) -> bool:
        return self._state == 'pristine'

    def dirty(self) -> bool:
        return self._state == 'dirty'

    def resolving(self) -> bool:
        return self._state == 'resolving'

    def resolved(self) -> bool:
        return self._state == 'resolved'

    def moving(self) -> bool:
        return self._state == 'moving'


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
