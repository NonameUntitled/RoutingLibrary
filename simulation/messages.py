from functools import total_ordering
from typing import *


@total_ordering
class Package:
    def __init__(self, pkg_id, size, dst, start_time, contents):
        self.id = pkg_id
        self.size = size
        self.dst = dst
        self.start_time = start_time
        self.contents = contents
        self.node_path = []

    def __str__(self):
        return '{}#{}{}'.format(self.__class__.__name__, self.id,
                                str((self.dst, self.size, self.start_time, self.contents)))

    def __repr__(self):
        return '<{}>'.format(str(self))

    def __hash__(self):
        return hash((self.id, self.contents))

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id


class WorldEvent:
    """
    Utility class, which allows access to arbitrary attrs defined
    at object creation. Base class for `Message` and `Action`.
    """

    def __init__(self, **kwargs):
        self.contents = kwargs

    def __str__(self):
        return '{}: {}'.format(self.__class__.__name__, str(self.contents))

    def __repr__(self):
        return '<{}>'.format(str(self))

    def __getattr__(self, name):
        try:
            return super().__getattribute__('contents')[name]
        except KeyError:
            raise AttributeError(name)

    def getContents(self):
        return self.contents


class Bag(Package):
    def __init__(self, bag_id, dst, start_time, contents):
        super().__init__(bag_id, 0, dst, start_time, contents)
        self.last_conveyor = -1


class BagAppearanceEvent(WorldEvent):
    def __init__(self, src_id: int, bag: Bag):
        super().__init__(src_id=src_id, bag=bag)


class UnsupportedEventType(Exception):
    """
    Exception which is thrown by event handlers on encounter of
    unknown event type
    """
    pass


AgentId = Tuple[str, int]
