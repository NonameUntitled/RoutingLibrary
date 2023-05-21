from typing import Callable, Dict, Optional, List

class EventSeries:
    def __init__(self):
        self._entries = []

    def logEvent(self, time, value):
        self._entries.append({"time": time, "value": value})

    def countPerTime(self):
        count = 0
        computed_entries = []
        for entry in self._entries:
            count += 1
            computed_entries.append({"time": entry["time"], "value": count})

    def averagePerTimePeriod(self, period):
        from_time = 0
        count = 0
        sum = 0
        min = float('inf')
        max = float('-inf')

        computed_entries = []
        for entry in self._entries:
            if entry["time"] > from_time + period:
                computed_entries.append({"time": from_time, "value": sum / count, "min": min, "max": max})
                from_time += period
                count = 0
                min = float('inf')
                max = float('-inf')
            count += 1
            sum += entry["value"]
            if entry["value"] < min:
                min = entry["value"]
            if entry["value"] > max:
                max = entry["value"]
        return computed_entries

    def weighedAveragePerTimePeriod(self, period):
        from_time = 0
        count = 0
        sum = 0

        computed_entries = []
        for entry in self._entries:
            if entry["time"] > from_time + period:
                computed_entries.append({"time": from_time, "value": sum / count})
                from_time += period
                count = 0
                min = float('inf')
                max = float('-inf')
            count += 1
            sum += entry["value"]
            if entry["value"] < min:
                min = entry["value"]
            if entry["value"] > max:
                max = entry["value"]
        return computed_entries





class MultiEventSeries():
    def __init__(self, series: Dict[str, EventSeries]):
        self.series = series

    def logEvent(self, tag: str, time, value):
        self.series[tag].logEvent(time, value)