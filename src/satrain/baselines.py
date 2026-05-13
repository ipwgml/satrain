"""
satrain.baselines
=================

This module provides access to results from baseline retrievals.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import xarray as xr

from satrain.definitions import BASE_SENSORS, DOMAINS


BASELINES_GMI = {
    "era5": "ERA5",
    "gprof_v07": "GPROF V07 (GMI)",
    "persiann_ccs": "PERSIANN CCS",
    "persiann_pdir_now": "PERSIANN PDIR-Now",
    "punet": "PU-Net",
    "gprof_ir": "GPROF-IR",
}


BASELINES_ATMS = {
    "era5": "ERA5",
    "gprof_v07": "GPROF V07 (ATMS)",
    "persiann_ccs": "PERSIANN CCS",
    "persiann_pdir_now": "PERSIANN PDIR-Now",
    "punet": "PU-Net",
    "gprof_ir": "GPROF-IR",
}


QPE_METRICS = {
    "bias": "Bias",
    "mae": "Mean Absolute Error",
    "mse": "Mean Squared Error",
    "smape": "Symmetric Mean Percentage Error",
    "correlation_coef": "Correlation Coef.",
}


DETECTION_METRICS = {
    "pod": "Probability of Detection",
    "far": "False Alarm Rate",
    "hss": "Heidke Skill Score",
    "area_under_curve": "Area under PR-Curve",
}


HEAVY_DETECTION_METRICS = {
    "pod_heavy": "Probability of Detection",
    "far_heavy": "False Alarm Rate",
    "hss_heavy": "Heidke Skill Score",
    "area_under_curve_heavy": "Area under PR-Curve",
}


ALL_METRICS = QPE_METRICS | DETECTION_METRICS | HEAVY_DETECTION_METRICS


FLIPPED_METRICS = [
    "correlcation_coef",
    "pod",
    "pod_heavy",
    "hss",
    "hss_heavy",
    "area_under_curve",
    "area_under_curve_heavy",
]

ABSOLUTE_METRICS = [
    "bias"
]


UNITS = {
    "bias": "%",
    "mae" : "[mm h⁻¹]",
    "mse" : "[(mm h⁻¹)²]",
    "smape" : "%",
}


RETRIEVAL_TYPES = {
    "GPROF-IR": "IR",
    "PERSIANN CCS": "IR",
    "ERA5": "Reanalysis",
    "GPROF V07 (GMI)": "PMW",
    "GPROF V07 (ATMS)": "PMW",
    "GPROF V08 (GMI)": "PMW",
    "GPROF V08 (ATMS)": "PMW",
    "PERSIANN PDIR-Now": "IR",
    "PU-Net": "IR"
}


def load_baseline_results(
    base_sensor: str,
    domain: str = "conus",
    baselines: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Load baseline results.

    Args:
        baselines: An optional list containing the baseline names to load. If not givne,
            results from all baselines are loaded.

    Return:
        An xarray.Dataset containing the merged baseline results.
    """
    if base_sensor.lower() == "gmi":
        BASELINES = BASELINES_GMI
    elif base_sensor.lower() == "atms":
        BASELINES = BASELINES_ATMS
    else:
        raise ValueError(
            f"Unsupport base sensor '{base_sensor}'. Shoule be one of {BASE_SENSORS}"
        )

    if baselines is None:
        baselines = BASELINES.keys()

    data_path = Path(__file__).parent / "files" / "baselines"
    results = []

    for baseline in baselines:
        if baseline not in BASELINES:
            raise ValueError(f"Encountered unsupported baseline name '{baseline}'.")

        results.append(xr.load_dataset(data_path / (f"{baseline}_{base_sensor.lower()}_{domain}.nc")))

    results = xr.concat(results, dim="algorithm")
    results["algorithm"] = (("algorithm",), [BASELINES[name] for name in baselines])
    return results


def get_ranked_results(
        base_sensor: str,
        metrics: list[str]
) -> "pd.Dataframe":
    """
    Loads baseline results in a pd.Dataframe sorted by their average rank across domains.

    Args:
        base_sensor: The name of the base sensor ('gmi' or 'atms').
        metrics: The names of the metrics to load.

    Return:
        A long-form pandas dataframe holding the baseline results.
    """
    import pandas as pd
    all_results = []
    ranks = []
    algorithms = None

    for domain in ["Austria", "CONUS", "Korea"]:
        results = load_baseline_results(base_sensor=base_sensor, domain=domain.lower())
        algorithms = results.algorithm.data
        results["kind"] = (("algorithm",), [RETRIEVAL_TYPES[alg] for alg in algorithms])
        res_d = results[list(metrics) + ["kind"]].to_dataframe()
        res_d["domain"] = domain
        ranks_m = []

        for metric in metrics:
            stat = results[metric].data
            if metric in ABSOLUTE_METRICS:
                stat = np.abs(stat)
            rank = np.argsort(stat)
            if metric in FLIPPED_METRICS:
                rank = np.flip(rank)
            rank = np.argsort(rank).astype(np.float32)
            rank[np.isnan(stat)] = np.nan
            ranks_m.append(rank)
        ranks.append(ranks_m)

        all_results.append(res_d)

    ranks = np.array(ranks)
    algorithms = list(algorithms[np.argsort(np.nanmean(ranks, (0, 1)))])
    all_results = pd.concat(all_results)

    dframe = (
        all_results.reset_index()
        .melt(
            id_vars=["algorithm", "domain", "kind"],
            value_vars=metrics,
            var_name="metric",
            value_name="value",
        )
    )

    all_ranks = np.array([algorithms.index(alg) for alg in dframe.algorithm])
    dframe["rank"] = all_ranks

    return dframe.sort_values(by=["domain", "rank"])


def plot_baselines_interactive(
        results: "pd.Dataframe",
        add_menu: bool = True,
) -> "plotly.Figure":
    """
    Visualizes baseline results using plotly for interactive display on the web.

    Args:
        results: A pandas long-form Dataframe holding the results to display.
        add_menu: If 'True', a menu is added to subset the displayed algorithms.

    Return:
        The plotly figure object.
    """
    import numpy as np
    import plotly.express as px
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns

    algorithms = list(np.unique(results.algorithm))

    cmap = plt.get_cmap("Set2")
    cmap = plt.get_cmap("flare")
    colors = [
        mcolors.to_hex(cmap(x))
        for x in np.linspace(0.1, 0.9, len(algorithms))
    ]

    metrics = {name: value for name, value in ALL_METRICS.items() if name in np.array(results.metric)}

    fig = px.bar(
        results,
        x="domain",
        y="value",
        color="algorithm",
        facet_col="metric",
        barmode="group",
        height=400,
        width=1200,
        labels=metrics.keys(),
        facet_col_spacing=0.06,
        category_orders={
            "metric": metrics,
            "domain": DOMAINS
        },
        hover_data=[
            "domain",
            "metric",
            "algorithm",
            "kind"
        ],
        color_discrete_sequence=colors
    )
    fontsize = 12
    for ind, (key, name) in enumerate(metrics.items()):
        unit = UNITS.get(key, "")
        fig.update_yaxes(title_text=f"{name} {unit}", col=ind + 1, showticklabels=True)

    def format_title(annotation):
        text = annotation.text
        if text.startswith("metric="):
            key = text.split("=")[1]
            annotation.update(
                text=metrics[key],
                font=dict(size=fontsize)
            )
    fig.for_each_annotation(format_title)
    fig.update_layout(
        legend=dict(
            title="Algorithm",
            orientation="h",
            yanchor="top",
            y=-0.4,
            xanchor="center",
            x=0.5,
        )
    )
    fig.update_xaxes(tickangle=45, title="Domain", title_font=dict(size=fontsize))
    fig.update_yaxes(matches=None, title_font=dict(size=fontsize))
    fig.update_layout(showlegend=True)
    fig.update_layout(
        font=dict(size=fontsize)
    )

    # Menu buttons.
    if add_menu:
        kinds = ["All"] + sorted(results["kind"].dropna().unique())
        buttons = []

        for t in kinds:
            if t == "All":
                visible = [True] * len(fig.data)
            else:
                visible = [
                    trace.name in results.loc[results["kind"] == t, "algorithm"].unique()
                    for trace in fig.data
                ]

            buttons.append(
                dict(
                    label=t,
                    method="update",
                    args=[
                        {"visible": visible},
                        {"title": f"Algorithms: {t}"},
                    ],
                )
            )

        fig.add_annotation(
            text="<b>Product Type:</b>",
            x=0.00,
            y=1.38,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            font=dict(size=14),
            xanchor="left",
            yanchor="top",
        )

        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    x=0.12,
                    y=1.4,
                    xanchor="left",
                    yanchor="top",
                )
            ]
        )
    return fig
