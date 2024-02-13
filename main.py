"""Generate BoxPlots from the stress-ng executions."""

import os
import re
from enum import IntEnum
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Fixing random state for reproducibility
np.random.seed(19680801)


class StressNGHeaders(IntEnum):
    """Keeps order of columns reported by stress-ng"""
    STRESSOR = 0
    BOGO_OPS_S_REAL_TIME = 5


def list_reports(directory) -> list:
    """return the list of files to be parsed"""
    search_dir = os.path.join('data', directory)
    reports = []
    for _, _, files in os.walk(search_dir, topdown=False):
        for name in files:
            reports.append(os.path.join(search_dir, name))
    return reports


def read_file(file) -> str:
    """returns the raw strings of a file if the process has read access"""
    return_value = None
    if os.access(file, os.R_OK):
        with open(file, encoding='utf8') as fp:
            return_value = fp.read()
    return return_value


def read_content(report) -> dict:
    """read a report and return a map of results per operation"""
    bogos = dict()

    content = read_file(report)
    metrics = [
        METRIC_PATTERN.sub('', l) for l in content.splitlines() if METRIC_PATTERN.search(l)
    ]

    if not metrics:
        return bogos

    # The first two elements account for the header and the units
    # stressor,"bogo ops","real time","usr time","sys time","bogo ops/s (real time)","bogo ops/s (usr+sys time)"
    for metric in metrics[2:]:
        columns = re.split(r'\s+', metric)
        s = columns[StressNGHeaders.STRESSOR]
        ops = float(columns[StressNGHeaders.BOGO_OPS_S_REAL_TIME])
        bogos.update({s: ops})

    return bogos


METRIC_PATTERN = re.compile(r'stress-ng: metrc: \[\d+\]\s+')

if __name__ == "__main__":

    carrier = dict()

    for configuration in ['SCHED_OTHER', 'SCHED_AUTOGROUP']:
        # Initiallize per SCHED
        carrier.update({configuration: {}})
        for report in list_reports(configuration):
            data = read_content(report)
            for stressor, v in data.items():
                if stressor not in carrier[configuration]:
                    carrier[configuration][stressor] = pd.DataFrame()
                carrier[configuration][stressor].append(v)
    print(carrier)

    # Graph
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    # generate some random test data
    for i, sched in enumerate(carrier):
        all_data = [v for _,v in carrier[sched].items()]
        # plot violin plot
        axs[i].boxplot(all_data)
        axs[i].set_title(sched)

        # adding horizontal grid lines
        stressors = [k for k,_ in carrier[sched].items()]

    for ax in axs:
        ax.yaxis.grid(True)
        ax.set_xticks([y + 1 for y in range(len(all_data))], labels=stressors, rotation=45)
        ax.set_xlabel('Stressors')
        ax.set_ylabel('Bogo ops/s (real time)')

    plt.show()

    # TODO: https://www.geeksforgeeks.org/data-normalization-with-pandas/