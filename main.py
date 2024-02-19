"""Generate BoxPlots from the stress-ng executions."""

import itertools
import os
import re
from matplotlib import pyplot as plt
import pandas as pd

from lib.utils import PROCCESORS, CLASS_CPU, SCHEDULERS, StressNGHeaders
from lib.utils import METRIC_PATTERN


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
    # cpu_stressors = [m for m in m for m in metrics[2:] if m in CLASS_CPU if m in CLASS_CPU]
    for metric in metrics[2:]:
        columns = re.split(r'\s+', metric)
        stressor = columns[StressNGHeaders.STRESSOR]
        ops = float(columns[StressNGHeaders.BOGO_OPS_S_REAL_TIME])
        if stressor in CLASS_CPU:
            bogos.update({stressor: ops})
    return bogos


if __name__ == "__main__":
    results = dict()

    # Parse stress-ng results from files
    for processor, scheduler in itertools.product(PROCCESORS, SCHEDULERS):
        # Initiallize per chiip-sched
        if processor not in results:
            results.update({processor: {}})
        
        if scheduler not in results[processor]:
            results[processor].update({scheduler: {}})

        report_path = os.path.join(processor, scheduler)
        for report in list_reports(report_path):
            data = read_content(report)
            for stressor, v in data.items():
                if stressor not in results[processor][scheduler]:
                    results[processor][scheduler][stressor] = []
                results[processor][scheduler][stressor].append(v)

    # Normalize with Min-Max
    for processor, schedulers in results.items():
        for scheduder, stressors in schedulers.items():
            stressors_df = pd.DataFrame.from_dict(stressors)
            for s, values in stressors_df.items():
                stressors[s] = (values - values.min()) / (values.max() - values.min())

    # Graph
    n_proccessors = len(results.keys())
    n_schedulers = len(results.items())

    # fig, axs = plt.subplots(nrows=n_proccessors, ncols=n_schedulers, sharex=False, sharey=True)
    fig = plt.figure(constrained_layout=True)
    fig.suptitle('Stress-NG executions by Proccessor and Kernel configuration (Linux 6.1.59)')

    # create subfigs
    subfigs = fig.subfigures(nrows=n_proccessors, ncols=1)
    for row, sub_proccesor in enumerate(zip(subfigs,results)):
        subfig, proccesor = sub_proccesor
        subfig.suptitle(f'{PROCCESORS[proccesor]}')

        # create 1x2 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=n_schedulers)
        for col, ax_scheduder in enumerate(zip(axs,results[proccesor])):
            ax, scheduder = ax_scheduder
            ax.set_title(f'{scheduder}')
            all_data = list(results[proccesor][scheduder].values())
            ax.boxplot(all_data, showfliers=False)  # showfliers: Show the outliers beyond the caps.
            # X labels
            ax.set_xlabel('Stressors')
            stressors = list(results[proccesor][scheduder].keys())
            print(stressors)
            ax.set_xticks([y + 1 for y in range(len(all_data))], labels=stressors, rotation=45)
            # Y labels
            ax.set_ylabel('Normalized Bogo ops/s (real time)')
            ax.yaxis.grid(True)

    # Unpacakge by processor and scheduler options
    # for i, processor in enumerate(results):
    #     for j, scheduder in enumerate(results[processor]):
            # axs[i][j].boxplot(all_data, showfliers=False)  # showfliers: Show the outliers beyond the caps.
            # axs[i][j].set_title(f'{scheduder}')

    # adding horizontal grid lines
    labels = [f'S{i+1}' for i in range(len(results[processor][scheduder].values())) ]

    # # Add table vertical
    # columns = ('Name')
    # cell_text = []
    # n_rows = len(stressors)
    # for row in range(n_rows):
    #     cell_text.append([stressors[row]])

    # plt.table(
    #     cellText=cell_text,
    #     rowLabels=labels,
    #     colLabels=columns,
    #     loc='right'
    # )
    plt.show()