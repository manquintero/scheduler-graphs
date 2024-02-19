"""Generate BoxPlots from the stress-ng executions."""

import itertools
import os
from matplotlib import pyplot as plt
import pandas as pd

from lib.utils import PROCCESORS, SCHEDULERS
from lib.utils import list_reports
from lib.utils import read_content

# Show normalized data
is_normalized = True


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
    if is_normalized:
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
    fig.suptitle(
        "Stress-NG executions by Proccessor and Kernel configuration (Linux 6.1.59)"
    )

    # create subfigs
    subfigs = fig.subfigures(nrows=n_proccessors, ncols=1)
    for row, subfig_proccesor in enumerate(zip(subfigs, results)):
        subfig, proccesor = subfig_proccesor
        subfig.suptitle(f"{PROCCESORS[proccesor]}")

        # create 1x2 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=n_schedulers, sharey=True)
        for col, ax_scheduder in enumerate(zip(axs, results[proccesor])):
            ax, scheduder = ax_scheduder
            ax.set_title(f"{scheduder}")
            all_data = list(results[proccesor][scheduder].values())
            ax.boxplot(
                all_data, showfliers=False
            )  # showfliers: Show the outliers beyond the caps.
            # X labels
            ax.set_xlabel("Stressors")
            stressors = list(results[proccesor][scheduder].keys())
            ax.set_xticks(
                [y + 1 for y in range(len(all_data))], labels=stressors, rotation=45
            )
            # Y labels
            ax.set_ylabel(f"{'Normalized' if is_normalized else ''} Bogo ops/s (real time)")
            ax.yaxis.grid(True)

    plt.show()
