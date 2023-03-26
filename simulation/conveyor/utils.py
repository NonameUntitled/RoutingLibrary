from __future__ import annotations

from functools import total_ordering
from typing import *


@total_ordering
class Bag:
    def __init__(self, bag_id: int, dst_type: str, dst_id: int, start_time: int, contents: Dict[str, Any]):
        self._id = bag_id
        self._dst_type = dst_type
        self._dst_id = dst_id
        self._start_time = start_time
        self._checkpoint_time = start_time
        self._contents = contents
        self._node_path = []
        self._last_conveyor_id = -1

    def __str__(self):
        return '{}#{}{}'.format(self.__class__.__name__, self._id,
                                str((self._dst_type, self._dst_id, self._start_time, self._contents)))

    def __repr__(self):
        return '<{}>'.format(str(self))

    def __hash__(self):
        return hash((self._id, self._contents))

    def __eq__(self, other: Bag):
        return self._id == other._id

    def __lt__(self, other: Bag):
        return self._id < other._id

    @property
    def id(self):
        return self._id

    def check_point(self, current_time):
        self._checkpoint_time = current_time


class WorldEvent:
    """
    Utility class, which allows access to arbitrary attrs defined
    at object creation. Base class for `Message` and `Action`.
    """

    def __init__(self):
        self._contents = vars(self)

    def __str__(self):
        return '{}: {}'.format(self.__class__.__name__, str(self._contents))

    def __repr__(self):
        return '<{}>'.format(str(self))

    def __getattr__(self, name: str):
        try:
            return super().__getattribute__('_contents')[name]
        except KeyError:
            raise AttributeError(f'No such attribute: {name}. Available: {list(self._contents.keys())[:-1]}')

    def getContents(self):
        return self._contents


class BagAppearanceEvent(WorldEvent):
    def __init__(self, src_id: int, bag: Bag):
        self._src_id = src_id
        self._bag = bag
        super().__init__()


class UnsupportedEventType(Exception):
    """
    Exception which is thrown by event handlers on encounter of
    unknown event type
    """
    pass
