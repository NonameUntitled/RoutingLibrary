from typing import Callable, Dict, Optional, Any

import pandas as pd

Aggregator = Callable[[Optional[float], float], float]


class EventLog:  # TODO [Vladimir Baikalov]: Create common logger, add tensorboards

    def __init__(
            self,
            key: str,
            period: int,
            aggregators: Dict[str, Aggregator] = None
    ):
        self._key = key
        self._period = period
        self._aggregators = aggregators or {}
        self._fields = ['time'] + list(aggregators.keys())

        self._records = []

    def _log(self, event_period: int, value: Any):
        assert event_period <= len(self._records), '`event_period` should be not greater then `len(self._records)`'

        if event_period == len(self._records):
            new_record = {
                aggregator_name: value
                for aggregator_name in self._aggregators
            }
            self._records.append(new_record)
        else:
            for aggregator_name, aggregator in self._aggregators.items():
                self._records[event_period][aggregator_name] = aggregator(self._records[event_period][aggregator_name],
                                                                          value)

    def log_event(self, time: float, value: Any):
        event_period = int(time / self._period)
        self._log(event_period, value)

    def get_dataframe(
            self,
            add_avg=False,
            avg_col='avg',
            sum_col='sum',
            count_col='count'
    ):
        df = pd.DataFrame(self.records, columns=self.columns, index=self.record_periods)
        if add_avg:
            df[avg_col] = df[sum_col] / df[count_col]

        return df.sort_index().astype(float, copy=False)

    def reset(self):
        self._records = []

    def save(self, filepath):
        self.getSeries().to_csv(filepath, index=False)

    def max_period(self):
        return self._records[-1][0]


class EventSeries:
    def __init__(self, period: int, aggregators: Dict[str, Aggregator]):
        self.columns = ['time'] + list(aggregators.keys())

        self.records = []
        self.record_idx = {}
        self.record_periods = []

        self.period = period
        self.aggregators = aggregators
        self.last_logged = 0

    def _log(self, cur_period: int, value):
        try:
            idx = self.record_idx[cur_period]
        except KeyError:
            idx = len(self.records)
            avg_time = cur_period * self.period
            row = [None] * len(self.columns)
            row[0] = avg_time
            self.records.append(row)
            self.record_periods.append(cur_period)
            self.record_idx[cur_period] = idx

        for i in range(1, len(self.columns)):
            col = self.columns[i]
            agg = self.aggregators[col]
            self.records[idx][i] = agg(self.records[idx][i], value)

    def logEvent(self, time, value):
        cur_period = int(time // self.period)
        self._log(cur_period, value)

    def logUniformRange(self, start, end, coeff):
        start_period = int(start // self.period)
        end_period = int(end // self.period)

        if start_period == end_period:
            self._log(start_period, (end - start) * coeff)
        else:
            start_gap = self.period - (start % self.period)
            end_gap = end % self.period

            self._log(start_period, coeff * start_gap)
            for period in range(start_period + 1, end_period):
                self._log(period, coeff * self.period)
            self._log(end_period, coeff * end_gap)

    def getSeries(
            self,
            add_avg=False,
            avg_col='avg',
            sum_col='sum',
            count_col='count'
    ):
        df = pd.DataFrame(self.records, columns=self.columns, index=self.record_periods)
        if add_avg:
            df[avg_col] = df[sum_col] / df[count_col]

        return df.sort_index().astype(float, copy=False)

    def reset(self):
        self.records = self.records.iloc[0:0]

    def load(self, data):
        if type(data) == str:
            df = pd.read_csv(data, index_col=False)
        else:
            df = data
        self.records = df.values.tolist()
        self.record_periods = list(range(len(self.records)))
        self.record_idx = {i: i for i in range(len(self.records))}

    def save(self, csv_path):
        self.getSeries().to_csv(csv_path, index=False)

    def maxTime(self):
        return self.records[-1][0]
