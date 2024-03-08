"""Generate BoxPlots from the stress-ng executions."""

import itertools
import os
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np

from lib.utils import PROCESSOR, SCHEDULERS
from lib.utils import list_reports
from lib.utils import read_content

# Show normalized data
is_normalized = True


if __name__ == "__main__":
    results = dict()
    variations = dict()
    means = dict()

    # Parse stress-ng results from files
    for processor, scheduler in itertools.product(PROCESSOR, SCHEDULERS):
        # Initiallize per chip-sched
        if processor not in results:
            results.update({processor: {}})
            variations.update({processor: {}})
            means.update({processor: {}})

        if scheduler not in results[processor]:
            results[processor].update({scheduler: {}})
            variations[processor].update({scheduler: {}})
            means[processor].update({scheduler: {}})

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
                variations[processor][scheduder] = (
                    stressors_df.std() / stressors_df.mean()
                )
                means[processor][scheduder] = stressors_df.mean()
                for s, values in stressors_df.items():
                    stressors[s] = (values - values.min()) / (
                        values.max() - values.min()
                    )

    # Graph
    n_processors = len(results.keys())
    n_schedulers = len(results.items())

    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    fig.suptitle(
        "Ejecuciones de Stress-NG por Procesador y configuración de Kernel (Linux 6.1.59)"
    )

    # create subfigs
    subfigs = fig.subfigures(nrows=n_processors, ncols=1)
    for row, subfig_processor in enumerate(zip(subfigs, results)):
        subfig, processor = subfig_processor
        subfig.suptitle(f"{PROCESSOR[processor]}")

        # create 1x2 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=n_schedulers, sharey=False)
        for col, ax_scheduder in enumerate(zip(axs, results[processor])):
            ax, scheduder = ax_scheduder
            ax.set_title(f"{scheduder}")
            all_data = list(results[processor][scheduder].values())
            ax.boxplot(
                all_data, showfliers=False
            )  # showfliers: Show the outliers beyond the caps.
            # X labels
            ax.set_xlabel("Stressors")
            stressors = list(results[processor][scheduder].keys())
            ax.set_xticks(
                [y + 1 for y in range(len(all_data))], labels=stressors, rotation=45
            )
            # Y labels
            ax.set_ylabel(
                f"{'Normalized' if is_normalized else ''} Bogo ops/s (real time)"
            )
            ax.yaxis.grid(True)
    plt.savefig("images/ejecuciones.png")

    # Graficar coeficientes
    stressors = tuple(variations[processor][scheduder].keys())
    x = np.arange(len(stressors))
    width = 0.2
    multiplier = 0

    fig, ax = plt.subplots(figsize=(20, 10))
    for processor, schedulers in variations.items():
        for scheduder, stressors in schedulers.items():
            attribute = f"{scheduder} ({processor})"
            offset = width * multiplier
            porcentajes = 100 * round(stressors, 2)
            rects = ax.bar(x + offset, porcentajes, width, label=attribute)
            ax.bar_label(rects, padding=3)
            multiplier += 1

    # Información sobre ejes
    ax.set_title("Coeficiente de variación por prueba de estrés")
    ax.set_xlabel("Stressors")
    ax.set_xticks(x + width, labels=stressors.index, rotation=45)
    ax.set_ylabel("Coeficiente de Variación")
    ax.yaxis.grid(True)
    ax.legend(loc="upper left", ncols=2)
    plt.show()
    plt.savefig("images/coeficientes.png")

    # Graficar el resumen de valores
    fig, ax = plt.subplots(figsize=(20, 10))
    # Esconder ejes
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")

    # Tabla
    delta_template = "Cambio fraccionario ({})"
    means_df = pd.DataFrame()
    for processor, schedulers in means.items():
        # Keep track of the schedulers to calculat the delta
        scheduder_columns = []
        for scheduder, stressors in schedulers.items():
            attribute = f"{scheduder} ({processor})"
            means_df[attribute] = stressors
            scheduder_columns.append(attribute)
        attribute = delta_template.format(processor)
        means_df[attribute] = (
            (means_df[scheduder_columns[-1]] - means_df[scheduder_columns[0]])
            / means_df[scheduder_columns[0]]
            * 100
        )

    # Preparar colores para
    colors = []
    n_rows, n_col = means_df.shape
    index_processors = [
        (
            means_df.columns.get_loc(delta_template.format(processor)),
            delta_template.format(processor),
        )
        for processor in PROCESSOR
    ]
    for row_name, row in means_df.iterrows():
        colors_in_column = ("aliceblue " * n_col).split()
        for i, col_name in index_processors:
            colors_in_column[i] = "tomato" if row[col_name] < 0 else "lightgreen"
        colors.append(colors_in_column)

    # Graficar
    ax.table(
        cellText=means_df.astype(np.int32).to_numpy(),
        cellColours=colors,
        cellLoc="center",
        rowLabels=stressors.index,
        colLabels=means_df.columns,
        loc="center",
    )
    fig.tight_layout()
    plt.show()
    plt.savefig("images/tabla.png")
