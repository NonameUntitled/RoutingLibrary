import csv
import os
from typing import Callable, Dict, Optional, List

import pandas as pd
from matplotlib import pyplot as plt


class EventSeries:
    def __init__(self, name: str, aggregation: str, experiment_name: str):
        self._entries = []
        self._aggregation = aggregation
        self._name = name
        self._experiment_name = experiment_name

    def logEvent(self, time, value):
        self._entries.append({"time": time, "value": value})

    def count_per_time(self):
        count = 0
        computed_entries = []
        for entry in self._entries:
            count += 1
            computed_entries.append({"time": entry["time"], "value": count})
        return computed_entries

    def average_per_time_period(self, period):
        from_time = 0
        count = 0
        sum = 0
        min = float('inf')
        max = float('-inf')

        computed_entries = []
        cur_entry_idx = 0
        while cur_entry_idx < len(self._entries):
            entry = self._entries[cur_entry_idx]
            if entry["time"] > from_time + period:
                computed_entries.append({"time": from_time, "value": sum / count, "min": min, "max": max})
                from_time += period
                count = 0
                min = float('inf')
                max = float('-inf')
                sum = 0
                continue
            count += 1
            sum += entry["value"]
            if entry["value"] < min:
                min = entry["value"]
            if entry["value"] > max:
                max = entry["value"]
            cur_entry_idx += 1
        return computed_entries

    def weighted_average_per_time_period(self, period):
        from_time = 0
        sum = 0
        prev_time = 0

        computed_entries = []
        cur_entry_idx = 0
        while cur_entry_idx < len(self._entries):
            entry = self._entries[cur_entry_idx]
            if entry["time"] > from_time + period:
                sum += entry["value"] * ((from_time + period - prev_time) / period)
                prev_time = from_time + period
                computed_entries.append({"time": from_time, "value": sum})
                from_time += period
                sum = 0
                continue
            sum += entry["value"] * ((entry["time"] - prev_time) / period)
            prev_time = entry["time"]
            cur_entry_idx += 1
        return computed_entries

    def get_aggregated(self):
        if self._aggregation == "count":
            return self.count_per_time()
        elif self._aggregation == "average":
            return self.average_per_time_period(200)
        elif self._aggregation == "weighted_average":
            return self.weighted_average_per_time_period(200)
        else:
            raise Exception(f"Unknown aggregation type {self._aggregation}")

    def get_df(self):
        return pd.DataFrame(self.get_aggregated())

    def get_df_entries(self):
        return pd.DataFrame(self._entries)

    def write_to_csv(self, path_prefix):
        df = self.get_df()
        df.to_csv(f'{path_prefix}{self._name}_{self._experiment_name}.csv', index=False)

        df = self.get_df_entries()
        df.to_csv(f'{path_prefix}{self._name}_{self._experiment_name}_entries.csv', index=False)

    def save_series(self, path):
        self.write_to_csv(path)

    def draw(self, path):
        df = self.get_df()
        plt.figure(figsize=(12, 6))
        if not df.empty:
            plt.plot(df['time'], df['value'], color='red', label=self._experiment_name)
        plt.grid(True)
        plt.xlabel("time")
        plt.ylabel(self._name)
        plt.legend()
        plt.savefig(f'{path}{self._name}.png')
        plt.close()

    def draw_with_existing(self, path_prefix, existing_path):
        # TODO: color generator
        colors = ['gray', 'brown', 'pink', 'purple', 'orange', 'black', 'yellow', 'blue', 'green', 'red']
        self.save_series(path_prefix)
        all_files = os.listdir(existing_path)
        csv_files = [f for f in all_files if self._name in f and f.endswith('.csv')]

        plt.figure(figsize=(12, 6))

        for csv_file in csv_files:
            label = csv_file.split('_')[-1].split('.')[0]
            try:
                df = pd.read_csv(f'{existing_path}{csv_file}')
            except:
                plt.plot([], [], label=label, color=colors.pop())
                continue
            if df.empty:
                plt.plot([], [], label=label, color=colors.pop())
                continue
            if not df.empty:
                plt.plot(df['time'], df['value'], label=label, color=colors.pop())
        plt.grid(True)
        plt.xlabel("time")
        plt.ylabel(self._name)
        plt.legend()
        plt.savefig(f'{existing_path}all_{self._name}.png')
        plt.close()


class MultiEventSeries():
    def __init__(self, series: Dict[str, EventSeries]):
        self.series = series

    def logEvent(self, tag: str, time, value):
        self.series[tag].logEvent(time, value)

    def save_series(self, tag: str, path_prefix):
        self.series[tag].save_series(path_prefix)

    def save_all_series(self, path_prefix):
        for tag in self.series:
            self.save_series(tag, path_prefix)

    def draw_series(self, tag: str, path):
        self.series[tag].draw(path)

    def draw_all_series(self, path):
        for tag in self.series:
            self.draw_series(tag, path)

    def draw_series_with_existing(self, tag: str, path_prefix, existing_path):
        self.series[tag].draw_with_existing(path_prefix, existing_path)

    def draw_all_series_with_existing(self, path_prefix, existing_path):
        for tag in self.series:
            self.series[tag].draw_with_existing(path_prefix, existing_path)
