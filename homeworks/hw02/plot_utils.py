
from pathlib import Path

import matplotlib.pyplot as plt


def plot_pressure_over_time(table, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        table["time_h"],
        table["boundary_pressure_mpa"],
        color="blue",
        marker="o",
        label="Давление на границе",
    )

    ax.plot(
        table["time_h"],
        table["well_pressure_mpa"],
        color="red",
        marker="s",
        label="Давление в скважине",
    )

    min_index = table["well_pressure_mpa"].idxmin()
    min_time = table.loc[min_index, "time_h"]
    min_pressure = table.loc[min_index, "well_pressure_mpa"]

    ax.annotate(
        f"Минимум: {min_pressure:.2f} МПа",
        xy=(min_time, min_pressure),
        xytext=(15, -25),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )

    ax.set_xlabel("Время, ч")
    ax.set_ylabel("Давление, МПа")
    ax.set_title("Изменение давления во времени")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    output_dir = Path(output_dir)
    fig.savefig(
        output_dir / "pressure_over_time.png",
        dpi=200,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "pressure_over_time.svg",
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_pressure_by_radius(table, output_dir):
    wells_sorted = table.sort_values("radius_m")

    assert (wells_sorted["radius_m"] > 0).all()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        wells_sorted["radius_m"],
        wells_sorted["pressure_mpa"],
        linestyle="none",
        marker="o",
        markersize=8,
        color="green",
        label="Наблюдательные скважины",
    )

    ax.set_xscale("log")

    ax.set_xlabel("Расстояние, м")
    ax.set_ylabel("Давление, МПа")
    ax.set_title("Давление в наблюдательных скважинах")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    output_dir = Path(output_dir)
    fig.savefig(
        output_dir / "pressure_by_radius.png",
        dpi=200,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "pressure_by_radius.svg",
        bbox_inches="tight",
    )

    plt.close(fig)
